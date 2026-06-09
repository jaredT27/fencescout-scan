"""Jefferson County, KY (Louisville/LOJIC) parcel adapter.

LOJIC = Louisville/Jefferson County Information Consortium. Free, public REST.
Parcel polygons + addresses joinable via PARCELID. No assessed/market values
in the public service — those live in jeffersonpva.ky.gov which doesn't have
an open feature service. We pull addresses + polygons here; value enrichment
is a downstream concern (Census ACS, ATTOM, Zillow scrape).
"""
import requests, math
from typing import Optional


PARCEL_URL = "https://gis.lojic.org/maps/rest/services/LojicSolutions/OpenDataPVA/MapServer/1/query"
ADDRESS_URL = "https://gis.lojic.org/maps/rest/services/LojicSolutions/OpenDataAddresses/MapServer/0/query"

ADDRESS_FIELDS = "ADDRESS,PARCELID,HOUSENO,DIR,STRNAME,TYPE,ZIPCODE,SIFCODE"


def _build_address(a: dict) -> str:
    parts = [str(a.get("HOUSENO", "")), (a.get("DIR") or "").strip(), (a.get("STRNAME") or "").strip(), (a.get("TYPE") or "").strip()]
    return " ".join(p for p in parts if p)


def _polygon_perimeter_ft_haversine(ring: list) -> int:
    R = 6_371_000
    total = 0.0
    pts = ring + [ring[0]] if ring and ring[0] != ring[-1] else ring
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        lat1, lat2 = math.radians(y1), math.radians(y2)
        dlat, dlng = math.radians(y2 - y1), math.radians(x2 - x1)
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
        total += 2 * R * math.asin(min(1.0, math.sqrt(a)))
    return int(total * 3.281)


def _fetch_parcel_geom(parcel_id: str) -> Optional[list]:
    r = requests.get(
        PARCEL_URL,
        params={
            "where": f"PARCELID = '{parcel_id}'",
            "outFields": "PARCELID,SHAPE.LEN,SHAPE.AREA",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "json",
        },
        timeout=15,
    )
    r.raise_for_status()
    feats = r.json().get("features", [])
    if not feats:
        return None
    f = feats[0]
    rings = f.get("geometry", {}).get("rings", [])
    return rings[0] if rings else None


def _parse_address_feature(feat: dict, with_geom: bool = True) -> dict:
    a = feat["attributes"]
    pin = a.get("PARCELID")
    addr = _build_address(a)
    pt = feat.get("geometry", {})  # {x: lng, y: lat} for points
    poly = _fetch_parcel_geom(pin) if pin and with_geom else None
    perim_ft = _polygon_perimeter_ft_haversine(poly) if poly else None

    return {
        "address": addr,
        "municipality": "Louisville",
        "state": "KY",
        "zip": a.get("ZIPCODE"),
        "pams_pin": pin,
        "prop_class": "2",
        "land_value": None,
        "improvement_value": None,
        "assessed_value": None,
        "market_estimate": None,           # No public values from LOJIC
        "market_source": None,
        "last_sale_price": None,
        "last_sale_date": None,
        "year_built": None,
        "lot_acres": None,
        "bldg_desc": None,
        "polygon_lnglat": poly or [],
        "perimeter_ft": perim_ft,
        "address_point": (pt.get("x"), pt.get("y")) if pt else None,
    }


def lookup_by_address(street: str, municipality: str = "Louisville") -> Optional[dict]:
    """Find by street address. Tolerant text match against HOUSENO + STRNAME."""
    s = street.strip().upper()
    parts = s.replace(",", "").split()
    if not parts:
        return None
    house = parts[0]
    if not house.isdigit():
        return None
    rest = " ".join(parts[1:])
    where = f"HOUSENO = {int(house)} AND STRNAME LIKE '%{rest.split()[0]}%'"
    r = requests.get(
        ADDRESS_URL,
        params={
            "where": where,
            "outFields": ADDRESS_FIELDS,
            "returnGeometry": "true",
            "outSR": "4326",
            "resultRecordCount": "5",
            "f": "json",
        },
        timeout=15,
    )
    r.raise_for_status()
    feats = r.json().get("features", [])
    if not feats:
        return None
    return _parse_address_feature(feats[0])


def lookup_by_point(lat: float, lng: float) -> Optional[dict]:
    """Find nearest address point to (lat, lng) within 50m."""
    r = requests.get(
        ADDRESS_URL,
        params={
            "geometry": f"{lng},{lat}",
            "geometryType": "esriGeometryPoint",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "distance": "50",
            "units": "esriSRUnit_Meter",
            "outFields": ADDRESS_FIELDS,
            "returnGeometry": "true",
            "outSR": "4326",
            "resultRecordCount": "1",
            "f": "json",
        },
        timeout=15,
    )
    r.raise_for_status()
    feats = r.json().get("features", [])
    return _parse_address_feature(feats[0]) if feats else None


def list_zip(zip_code: str, prop_class: str = None, min_value: int = 0,
             limit: int = 600, with_polygon: bool = False) -> list[dict]:
    """List residential addresses in a Louisville-area ZIP.

    Filters out condos/multi-unit (anything with an APT field, or sharing a
    parcel ID with another address). Returns address + point geometry; polygon
    is fetched lazily.
    """
    where = f"ZIPCODE = '{zip_code}' AND HOUSENO > 0"
    r = requests.get(
        ADDRESS_URL,
        params={
            "where": where,
            "outFields": "ADDRESS,PARCELID,HOUSENO,DIR,STRNAME,TYPE,ZIPCODE,SIFCODE,APT",
            "returnGeometry": "true",
            "outSR": "4326",
            "resultRecordCount": str(limit),
            "f": "json",
        },
        timeout=30,
    )
    r.raise_for_status()
    feats = r.json().get("features", [])

    # Bucket by PARCELID. Multi-unit / townhouse parcels have many addresses
    # mapping to one PIN — skip those (Vince doesn't fence shared parcels).
    by_pin: dict[str, list] = {}
    for f in feats:
        pin = f["attributes"].get("PARCELID")
        if not pin:
            continue
        by_pin.setdefault(pin, []).append(f)

    parcels = []
    for pin, group in by_pin.items():
        if len(group) > 1:
            continue  # multi-unit
        f = group[0]
        a = f["attributes"]
        if (a.get("APT") or "").strip():  # named apartment unit
            continue
        try:
            p = _parse_address_feature(f, with_geom=with_polygon)
            parcels.append(p)
        except Exception:
            continue
    return parcels


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) >= 2 and sys.argv[1].isdigit():
        ps = list_zip(sys.argv[1], limit=15)
        print(f"found {len(ps)} parcels")
        for p in ps[:10]:
            print(f"  {p['address']:40s} {p['zip']}  perim={p['perimeter_ft']}ft  pin={p['pams_pin']}")
    else:
        p = lookup_by_address(sys.argv[1] if len(sys.argv) > 1 else "5108 Oakbrook")
        print(json.dumps({k: v for k, v in p.items() if k != "polygon_lnglat"}, indent=2) if p else "not found")
