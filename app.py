"""FenceScout scan service (Railway) — homes-list mode.

The browser sends the list of homes it already fetched from LOJIC (an IP LOJIC
allows). This service does ONLY the part that must run on a server: pull the
Google satellite per home and run OpenCV fence detection. Ranks by NO-FENCE
score (high = genuinely needs a fence = mail a postcard).

Endpoints:
  POST /scan       {homes:[{addr,lat,lng,pin?}], name?}  -> {job_id, home_count}
  GET  /scan/{id}                                        -> {status,total,scanned,results[]}
  GET  /selftest                                         -> {satellite: bool}  (Railway->Google check)
  GET  /                                                 -> health

Env: DATABASE_URL, ALLOW_ORIGINS, SCAN_API_KEY (optional), GOOGLE_MAPS_KEY (optional).
"""
import os, io, hashlib, time
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image
from fastapi import FastAPI, BackgroundTasks, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2, psycopg2.extras

import detect, imagery

RATE_WOOD, RATE_VINYL = 36, 49
DEFAULT_LF = 250                      # estimate until per-parcel perimeter is wired
DATABASE_URL = os.environ["DATABASE_URL"]
ALLOW_ORIGINS = os.environ.get("ALLOW_ORIGINS", "*").split(",")
SCAN_API_KEY = os.environ.get("SCAN_API_KEY", "")
MAPS_KEY = os.environ.get("GOOGLE_MAPS_KEY", "")
MAX_HOMES = int(os.environ.get("MAX_HOMES", "1500"))
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


class Home(BaseModel):
    addr: str
    lat: float
    lng: float
    pin: str | None = None


class ScanReq(BaseModel):
    homes: list[Home]
    name: str | None = None


def _require_key(x_api_key):
    if SCAN_API_KEY and x_api_key != SCAN_API_KEY:
        raise HTTPException(401, "bad api key")


def _sat_static(lat, lng):
    r = requests.get("https://maps.googleapis.com/maps/api/staticmap", params={
        "center": f"{lat},{lng}", "zoom": "20", "size": "600x600",
        "maptype": "satellite", "key": MAPS_KEY}, timeout=20)
    if r.status_code != 200 or len(r.content) < 2000:
        return None
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def get_sat(lat, lng):
    if MAPS_KEY:
        im = _sat_static(lat, lng)
        if im is not None:
            return im
    return imagery.pull_satellite(lat, lng, zoom=21, px_size=600)


def score_home(h):
    try:
        sat = get_sat(h["lat"], h["lng"])
        if sat is None:
            return None
        sc = detect.fence_score(sat, polygon_lnglat=None, center_latlng=(h["lat"], h["lng"]))
        nf = round(100 - sc["fence_present_score"], 1)
        lf = DEFAULT_LF
        return (h.get("pin") or "", h["addr"], h["lat"], h["lng"], nf,
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
    homes = [h.dict() for h in req.homes]
    if not homes:
        raise HTTPException(400, "no homes provided")
    if len(homes) > MAX_HOMES:
        raise HTTPException(413, f"{len(homes)} homes exceeds the {MAX_HOMES} cap")
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


@app.get("/selftest")
def selftest():
    """Confirm Railway can reach Google satellite (LOJIC blocked Railway, so verify Google)."""
    try:
        im = get_sat(38.28536, -85.48754)
        return {"satellite": im is not None, "size": (list(im.size) if im else None)}
    except Exception as e:
        return {"satellite": False, "error": str(e)[:200]}


@app.get("/")
def health():
    return {"ok": True, "service": "fencescout-scan"}
