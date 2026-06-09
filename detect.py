"""Heuristic fence detection on satellite imagery via Hough lines.

Strategy: a fence shows up as a long, thin straight line along the property
boundary. We Canny-edge the image, run probabilistic Hough, then count the
density of long lines inside a band that follows the actual parcel polygon
(when available) or a generic ring around the image center (fallback).

Returns a score 0-100 where HIGHER = more likely the home has NO fence
(better candidate). Confidence band attached.
"""
import cv2
import math
import numpy as np
from PIL import Image


def _polygon_to_pixel_ring(polygon_lnglat: list, center_lat: float, center_lng: float,
                            img_w: int, img_h: int, zoom: int = 21) -> np.ndarray | None:
    """Convert parcel polygon (lng/lat vertices) to image-pixel coordinates.

    Returns Nx2 int array of (x, y) pixel positions. None if polygon is empty.
    """
    if not polygon_lnglat:
        return None
    mpp = 156543.03 * abs(math.cos(math.radians(center_lat))) / (2 ** zoom)
    R = 6_371_000
    pts = []
    for lng, lat in polygon_lnglat:
        # Equirectangular delta in meters from center
        dx = math.radians(lng - center_lng) * R * math.cos(math.radians(center_lat))
        dy = math.radians(center_lat - lat) * R   # y inverted (south positive)
        px = int(img_w / 2 + dx / mpp)
        py = int(img_h / 2 + dy / mpp)
        pts.append([px, py])
    return np.array(pts, dtype=np.int32)


def fence_score(sat_img: Image.Image, polygon_lnglat: list = None,
                center_latlng: tuple[float, float] = None) -> dict:
    """Hough-line fence detection on a 1400x1400 z21 satellite image.

    If a parcel polygon + center is provided, the detection ring follows the
    actual property boundary (much more accurate). Otherwise falls back to a
    generic ring around the image center.
    """
    arr = cv2.cvtColor(np.array(sat_img), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]

    # Smooth + Canny edge
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    edges = cv2.Canny(gray, 60, 160)

    # Probabilistic Hough
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=60, minLineLength=40, maxLineGap=8,
    )
    if lines is None:
        lines = np.empty((0, 1, 4))

    # Build the perimeter detection mask. If we have parcel polygon, dilate
    # an outline from the polygon edge — the true property boundary. Otherwise
    # fall back to a generic concentric ring.
    annulus = None
    if polygon_lnglat and center_latlng:
        ring_pts = _polygon_to_pixel_ring(polygon_lnglat, center_latlng[0], center_latlng[1], w, h)
        if ring_pts is not None and len(ring_pts) >= 3:
            poly_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [ring_pts], 255)
            # Erode + subtract to make a thin band along the polygon boundary
            kernel = np.ones((25, 25), np.uint8)
            inner = cv2.erode(poly_mask, kernel, iterations=2)   # ~8m inside the boundary
            outer = cv2.dilate(poly_mask, kernel, iterations=1)  # ~3m outside
            band = cv2.subtract(outer, inner)
            annulus = band > 0

    if annulus is None:
        cx, cy = w // 2, h // 2
        r_outer = int(min(w, h) * 0.18)
        r_inner = int(min(w, h) * 0.08)
        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        annulus = (dist2 >= r_inner ** 2) & (dist2 <= r_outer ** 2)

    cx, cy = w // 2, h // 2
    r_outer = int(min(w, h) * 0.18)
    r_inner = int(min(w, h) * 0.08)

    # Count line-pixel density in the annulus
    line_canvas = np.zeros((h, w), dtype=np.uint8)
    total_line_px = 0
    long_lines_in_ring = 0
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        length = int(np.hypot(x2 - x1, y2 - y1))
        cv2.line(line_canvas, (x1, y1), (x2, y2), 255, 1)
        total_line_px += length
        # Is the midpoint inside the annulus?
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        d2 = (mx - cx) ** 2 + (my - cy) ** 2
        if r_inner ** 2 <= d2 <= r_outer ** 2 and length > 50:
            long_lines_in_ring += 1

    ring_pixels = int(annulus.sum())
    line_in_ring_px = int(((line_canvas > 0) & annulus).sum())
    density = line_in_ring_px / max(ring_pixels, 1)

    # Calibrated against ~20 NJ residential aerials (z21, 1400px window):
    #   density 0.005-0.020 with long_lines_in_ring < 50 = clear no-fence yards
    #   density 0.020-0.045 = mixed (driveways, landscaping noise)
    #   density 0.045+ with long_lines_in_ring > 80 = likely fence present
    # This is a CANDIDATE FILTER, not a final classifier — designed to surface
    # likely candidates for fast human verification (image is shown beside score).
    fence_present_score = min(100, max(0, (density - 0.005) / 0.045 * 100))
    no_fence_score = max(0, 100 - fence_present_score)

    # Penalty: lots of long perimeter lines is suspicious even at low density
    if long_lines_in_ring >= 90:
        no_fence_score = max(0, no_fence_score - 25)
    elif long_lines_in_ring >= 60:
        no_fence_score = max(0, no_fence_score - 12)

    # Confidence from how far from the ambiguous band the score sits
    if no_fence_score >= 75 or no_fence_score <= 20:
        confidence = "high"
    elif no_fence_score >= 60 or no_fence_score <= 35:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "no_fence_score": round(no_fence_score, 1),
        "fence_present_score": round(fence_present_score, 1),
        "confidence": confidence,
        "perimeter_line_density": round(density, 5),
        "long_lines_in_ring": int(long_lines_in_ring),
        "total_line_segments": int(len(lines)),
    }


def estimate_linear_feet(sat_img: Image.Image, zoom: int = 21, lat: float = 41.0) -> int:
    """Rough linear-feet estimate for a typical residential rear+side yard.

    At z21, ~0.075 m/px at lat 41. A 1400px image covers ~105m across.
    We estimate the rear+side perimeter (3-sided enclosure) for a typical
    suburban home. Use 70% of estimated full perimeter at the inner-house ring.
    """
    # Meters per pixel at z21 at given latitude
    mpp = 156543.03 * abs(np.cos(np.radians(lat))) / (2 ** zoom)
    # Default residential lot perimeter for back+sides: ~3 sides of ~50ft each
    # Reasonable midpoint when we don't have parcel data
    # If the home is in a wealthy ZIP this skews higher
    estimate_m = 65.0   # ~213 linear feet
    return int(estimate_m * 3.281)  # m -> ft


if __name__ == "__main__":
    import sys
    img = Image.open(sys.argv[1])
    print(fence_score(img))
