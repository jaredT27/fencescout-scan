"""Pull clean satellite + Street View imagery for an address."""
import math, io, os, re, sys, json
import requests
from PIL import Image

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://www.google.com/maps",
}
TILE_SIZE = 256


def geocode(address: str) -> tuple[float, float]:
    """Address -> (lat, lng). Uses OSM Nominatim (free, rate-limited 1/s)."""
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": address, "format": "json", "limit": 1},
        headers={"User-Agent": "fencescout/1.0"},
        timeout=15,
    )
    r.raise_for_status()
    j = r.json()
    if not j:
        raise ValueError(f"No geocode result for: {address}")
    return float(j[0]["lat"]), float(j[0]["lon"])


def latlng_to_tile(lat, lng, z):
    n = 2 ** z
    x = (lng + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


def pull_satellite(lat: float, lng: float, zoom: int = 21, px_size: int = 1400) -> Image.Image:
    """Stitch Google satellite tiles centered on (lat, lng)."""
    tx, ty = latlng_to_tile(lat, lng, zoom)
    cx, cy = tx * TILE_SIZE, ty * TILE_SIZE
    half = px_size // 2
    left_px, top_px = cx - half, cy - half
    x_start = int(math.floor(left_px / TILE_SIZE))
    x_end = int(math.floor((cx + half - 1) / TILE_SIZE))
    y_start = int(math.floor(top_px / TILE_SIZE))
    y_end = int(math.floor((cy + half - 1) / TILE_SIZE))
    canvas = Image.new("RGB", ((x_end - x_start + 1) * TILE_SIZE, (y_end - y_start + 1) * TILE_SIZE))
    for xi in range(x_start, x_end + 1):
        for yi in range(y_start, y_end + 1):
            url = f"https://mt1.google.com/vt/lyrs=s&x={xi}&y={yi}&z={zoom}"
            try:
                rr = requests.get(url, headers=HEADERS, timeout=15)
                rr.raise_for_status()
                t = Image.open(io.BytesIO(rr.content)).convert("RGB")
                canvas.paste(t, ((xi - x_start) * TILE_SIZE, (yi - y_start) * TILE_SIZE))
            except Exception as e:
                print(f"  tile fail z{zoom} x{xi} y{yi}: {e}", file=sys.stderr)
    off_x = left_px - x_start * TILE_SIZE
    off_y = top_px - y_start * TILE_SIZE
    return canvas.crop((int(off_x), int(off_y), int(off_x + px_size), int(off_y + px_size)))


def _find_panoid(html: str) -> tuple[str | None, float]:
    """Extract first plausible panoid + suggested yaw from a Google Maps HTML page."""
    # Most reliable: 22-char base64-ish tokens in the page state, picked by frequency
    tokens = re.findall(r'"([A-Za-z0-9_\-]{22})"', html)
    if not tokens:
        return None, 0.0
    # The actual panoid is repeated dozens of times in the page
    from collections import Counter
    panoid = Counter(tokens).most_common(1)[0][0]

    # Try to extract a yaw value near a streetviewpixels reference
    yaw = 0.0
    m = re.search(r"yaw=([\d.]+)", html)
    if m:
        try:
            yaw = float(m.group(1))
        except ValueError:
            pass
    return panoid, yaw


def pull_streetview(query: str = None, lat: float = None, lng: float = None) -> Image.Image | None:
    """Pull Google Street View thumbnail. Pass either an address string or lat/lng.

    The lat/lng path is more reliable. If only `query` is passed, we geocode first.
    """
    if lat is None or lng is None:
        if query is None:
            return None
        lat, lng = geocode(query)

    # Use the @lat,lng,3a,... URL form which forces Street View context in the page
    url = f"https://www.google.com/maps/@{lat},{lng},3a,75y,0h,90t/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    panoid, yaw = _find_panoid(r.text)
    if not panoid:
        return None

    thumb = (
        "https://streetviewpixels-pa.googleapis.com/v1/thumbnail"
        f"?cb_client=maps_sv.tactile&w=1920&h=1080&pitch=0&panoid={panoid}&yaw={yaw}"
    )
    rr = requests.get(thumb, headers=HEADERS, timeout=15)
    if rr.status_code != 200 or len(rr.content) < 5000:
        return None
    return Image.open(io.BytesIO(rr.content)).convert("RGB")


if __name__ == "__main__":
    addr = sys.argv[1] if len(sys.argv) > 1 else "23 Byrne Ln Harrington Park NJ 07640"
    lat, lng = geocode(addr)
    print(f"geocoded: {lat}, {lng}")
    sat = pull_satellite(lat, lng)
    sat.save("/tmp/test_sat.jpg", "JPEG", quality=92)
    print(f"sat saved: {sat.size}")
    sv = pull_streetview(addr)
    if sv:
        sv.save("/tmp/test_sv.jpg", "JPEG", quality=92)
        print(f"sv saved: {sv.size}")
    else:
        print("sv: not available")
