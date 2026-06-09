"""FenceScout scan service (Railway).

POST /scan      {polygon:[[lng,lat],...], name?}  -> {job_id, home_count}
GET  /scan/{id}                                   -> {status,total,scanned,results[]}
GET  /                                             -> health

Draws on the existing engine (parcel_ky, imagery, detect). For each home in the
drawn territory: parcel polygon -> centroid -> satellite -> detect.fence_score.
Ranks by NO-FENCE score (high = genuinely needs a fence = mail a postcard).

Env:
  DATABASE_URL     Railway Postgres (provided automatically)
  ALLOW_ORIGINS    comma list, e.g. https://top-rail-louisville.vercel.app
  SCAN_API_KEY     optional shared key required in X-Api-Key header
  GOOGLE_MAPS_KEY  optional; if set + Maps Static API allowed on the key, used
                   for satellite. Otherwise falls back to free tiles.
"""
import os, io, json, math, time, hashlib, threading
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image
from fastapi import FastAPI, BackgroundTasks, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2, psycopg2.extras

import detect, imagery, parcel_ky

ADDR_URL = "https://gis.lojic.org/maps/rest/services/LojicSolutions/OpenDataAddresses/MapServer/0/query"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
RATE_WOOD, RATE_VINYL, INSTALL_FRAC = 36, 49, 0.70
DATABASE_URL = os.environ["DATABASE_URL"]
ALLOW_ORIGINS = os.environ.get("ALLOW_ORIGINS", "*").split(",")
SCAN_API_KEY = os.environ.get("SCAN_API_KEY", "")
MAPS_KEY = os.environ.get("GOOGLE_MAPS_KEY", "")
MAX_HOMES = int(os.environ.get("MAX_HOMES", "1500"))   # guardrail against giant scans
WORKERS = int(os.environ.get("WORKERS", "8"))


def db():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    with db() as c, c.cursor() as cur:
        cur.execute("create table if not exists jobs(id text primary key, status text, "
                    "total int, scanned int default 0, created timestamptz default now())")
        cur.execute("create table if not exists results(job_id text, pin text, address text, "
                    "lat double precision, lng double precision, no_fence_score real, "
                    "has_fence boolean, lf int, quote_lo int, quote_hi int)")
        cur.execute("create index if not exists idx_results_job on results(job_id)")


init_db()
app = FastAPI(title="FenceScout Scan")
app.add_middleware(CORSMiddleware, allow_origins=ALLOW_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])


class ScanReq(BaseModel):
    polygon: list
    name: str | None = None


def _require_key(x_api_key):
    if SCAN_API_KEY and x_api_key != SCAN_API_KEY:
        raise HTTPException(401, "bad api key")


# ---------- LOJIC: homes inside the drawn polygon ----------
def homes_in_polygon(ring):
    ring = ring + [ring[0]]
    geom = json.dumps({"rings": [ring], "spatialReference": {"wkid": 4326}})
    feats, offset = [], 0
    for _ in range(8):
        r = requests.post(ADDR_URL, data={
            "geometry": geom, "geometryType": "esriGeometryPolygon", "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "ADDRESS,PARCELID,HOUSENO,STRNAME,TYPE,ZIPCODE,APT",
            "returnGeometry": "true", "outSR": "4326", "f": "json",
            "resultRecordCount": "2000", "resultOffset": str(offset),
        }, headers=UA, timeout=40).json()
        page = r.get("features", [])
        feats += page
        if not r.get("exceededTransferLimit") or not page:
            break
        offset += len(page)
    by_pin = {}
    for f in feats:
        pin = f["attributes"].get("PARCELID")
        if pin:
            by_pin.setdefault(pin, []).append(f)
    homes = []
    for pin, group in by_pin.items():
        if len(group) > 1:
            continue
        a = group[0]["attributes"]
        g = group[0].get("geometry", {})
        if (a.get("APT") or "").strip() or "x" not in g:
            continue
        addr = " ".join(p for p in [str(a.get("HOUSENO", "")).strip(),
                                    (a.get("STRNAME") or "").strip(),
                                    (a.get("TYPE") or "").strip()] if p)
        if not addr:
            continue
        homes.append({"pin": pin, "addr": addr, "lat": g["y"], "lng": g["x"]})
    return homes


def _centroid(ring):
    A = cx = cy = 0.0
    pts = ring + [ring[0]] if ring[0] != ring[-1] else ring
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        cr = x1 * y2 - x2 * y1
        A += cr; cx += (x1 + x2) * cr; cy += (y1 + y2) * cr
    if A == 0:
        xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    A *= 0.5
    return cx / (6 * A), cy / (6 * A)


def _sat_static(lat, lng):
    r = requests.get("https://maps.googleapis.com/maps/api/staticmap", params={
        "center": f"{lat},{lng}", "zoom": "20", "size": "600x600",
        "maptype": "satellite", "key": MAPS_KEY}, timeout=20)
    if r.status_code != 200 or len(r.content) < 2000:
        return None
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def _get_sat(lat, lng):
    if MAPS_KEY:
        im = _sat_static(lat, lng)
        if im is not None:
            return im
    return imagery.pull_satellite(lat, lng, zoom=21, px_size=600)   # free-tile fallback


def score_home(h):
    try:
        ring = parcel_ky._fetch_parcel_geom(h["pin"])
        if not ring:
            return None
        clng, clat = _centroid(ring)
        perim = parcel_ky._polygon_perimeter_ft_haversine(ring)
        if perim > 1200 or perim < 240:
            return None
        sat = _get_sat(clat, clng)
        if sat is None:
            return None
        sc = detect.fence_score(sat, polygon_lnglat=ring, center_latlng=(clat, clng))
        nf = round(100 - sc["fence_present_score"], 1)
        lf = round(perim * INSTALL_FRAC)
        return (h["pin"], h["addr"], h["lat"], h["lng"], nf,
                sc["fence_present_score"] >= 40, lf,
                round(lf * RATE_WOOD, -2), round(lf * RATE_VINYL, -2))
    except Exception:
        return None


def run_job(job_id, homes):
    def worker(h):
        res = score_home(h)
        conn = db()
        with conn, conn.cursor() as cur:
            if res:
                cur.execute("insert into results(job_id,pin,address,lat,lng,no_fence_score,"
                            "has_fence,lf,quote_lo,quote_hi) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                            (job_id,) + res)
            cur.execute("update jobs set scanned = scanned + 1 where id=%s", (job_id,))
        conn.close()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(worker, homes))
    conn = db()
    with conn, conn.cursor() as cur:
        cur.execute("update jobs set status='done' where id=%s", (job_id,))
    conn.close()


@app.post("/scan")
def scan(req: ScanReq, background: BackgroundTasks, x_api_key: str = Header(default="")):
    _require_key(x_api_key)
    if not req.polygon or len(req.polygon) < 3:
        raise HTTPException(400, "polygon needs >=3 points [[lng,lat],...]")
    homes = homes_in_polygon(req.polygon)
    if len(homes) > MAX_HOMES:
        raise HTTPException(413, f"{len(homes)} homes exceeds the {MAX_HOMES} cap — draw a smaller area")
    job_id = hashlib.sha1(f"{time.time()}:{len(homes)}".encode()).hexdigest()[:12]
    conn = db()
    with conn, conn.cursor() as cur:
        cur.execute("insert into jobs(id,status,total,scanned) values(%s,'running',%s,0)",
                    (job_id, len(homes)))
    conn.close()
    background.add_task(run_job, job_id, homes)
    return {"job_id": job_id, "home_count": len(homes)}


@app.get("/scan/{job_id}")
def status(job_id: str):
    conn = db()
    with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("select status,total,scanned from jobs where id=%s", (job_id,))
        job = cur.fetchone()
        if not job:
            raise HTTPException(404, "no such job")
        cur.execute("select address,lat,lng,no_fence_score,has_fence,lf,quote_lo,quote_hi "
                    "from results where job_id=%s order by no_fence_score desc", (job_id,))
        rows = cur.fetchall()
    conn.close()
    return {"status": job["status"], "total": job["total"], "scanned": job["scanned"],
            "results": rows}


@app.get("/")
def health():
    return {"ok": True, "service": "fencescout-scan"}
