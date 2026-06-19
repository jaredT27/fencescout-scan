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
import os, io, hashlib, time, json, threading
from datetime import datetime, timedelta, timezone
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

# --- ServiceMinder lead relay (Top Rail Fence) ---
# SM_LEAD_KEY is the addupdate API key from Jakob. Set it in Railway Variables;
# never hardcode it here. The key stays server-side so it never reaches the browser.
SM_LEAD_KEY = os.environ.get("SM_LEAD_KEY", "")
SM_LEAD_URL = "https://serviceminder.com/service/contact/addupdate/"
SM_CHANNEL = os.environ.get("SM_CHANNEL", "Direct Mail")    # their existing channel
SM_CAMPAIGN = os.environ.get("SM_CAMPAIGN", "Fence Scout")  # their identifier
SM_DEFAULT_STATE = os.environ.get("SM_DEFAULT_STATE", "KY")
# Pilot zips -> city, used to fill City when an address has no comma to split on.
ZIP_CITY = {"40245": "Louisville", "40059": "Prospect", "40223": "Louisville"}


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
        cur.execute("create table if not exists leads(id serial primary key, created timestamptz default now(), "
                    "name text, email text, phone text, address1 text, city text, state text, postal text, "
                    "variant text, page text, notes text, sm_result int, sm_contact_id text, sm_message text)")
        cur.execute("create table if not exists outbound_queue(id serial primary key, created timestamptz default now(), "
                    "send_after timestamptz, payload text, status text default 'queued', sent_at timestamptz, "
                    "sm_result int, sm_contact_id text, sm_message text)")
        cur.execute("create index if not exists idx_outbound_due on outbound_queue(status, send_after)")


init_db()
app = FastAPI(title="FenceScout Scan")
app.add_middleware(CORSMiddleware, allow_origins=ALLOW_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])


class Home(BaseModel):
    addr: str
    lat: float
    lng: float
    pin: str | None = None
    lf: int | None = None        # lineal feet from the browser's parcel measure; falls back to DEFAULT_LF


class ScanReq(BaseModel):
    homes: list[Home]
    name: str | None = None


def _require_key(x_api_key):
    if SCAN_API_KEY and x_api_key != SCAN_API_KEY:
        raise HTTPException(401, "bad api key")


# ---------------------------------------------------------------------------
# ServiceMinder lead relay
# ---------------------------------------------------------------------------
import re


class LeadReq(BaseModel):
    name: str | None = None
    address: str | None = None          # combined "123 Main St, Louisville, KY 40245"
    address1: str | None = None         # OR pass components directly (preferred)
    city: str | None = None
    state: str | None = None
    zip: str | None = None
    phone: str | None = None
    email: str | None = None
    version: str | None = None          # postcard variant a / b
    page: str | None = None             # landing URL
    lf: int | None = None               # estimated linear feet
    quote: str | None = None            # e.g. "$6,600 - $9,000"
    style: str | None = None            # fence style if captured
    test: bool | None = False           # mark a coordinated test lead


def _split_name(name):
    name = (name or "").strip()
    if not name:
        return "", ""
    parts = name.split()
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def _parse_address(combined):
    """Best-effort split of '123 Main St, Louisville, KY 40245' into parts.
    Returns (address1, city, state, postal). Missing pieces come back ''."""
    a1 = city = state = postal = ""
    s = (combined or "").strip()
    if not s:
        return a1, city, state, postal
    # pull "ST 12345" (state + zip) off the end wherever it sits
    m = re.search(r"\b([A-Za-z]{2})\s+(\d{5})(?:-\d{4})?\b", s)
    if m:
        state = m.group(1).upper()
        postal = m.group(2)
        s = (s[:m.start()] + s[m.end():]).strip().strip(",").strip()
    elif re.search(r"\b(\d{5})(?:-\d{4})?\b", s):  # bare zip
        z = re.search(r"\b(\d{5})(?:-\d{4})?\b", s)
        postal = z.group(1)
        s = (s[:z.start()] + s[z.end():]).strip().strip(",").strip()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) >= 2:
        a1 = parts[0]
        city = parts[1]
        if not state and len(parts) >= 3:
            state = parts[2].upper()
    elif len(parts) == 1:
        a1 = parts[0]
    return a1, city, state, postal


def _map_lead(d, source_note="Lead from FenceScout landing page."):
    """Map a lead dict -> ServiceMinder FORM-POST payload. Used by both the
    instant inbound /lead path and the delayed outbound flush."""
    first, last = _split_name(d.get("name"))
    a1, city, state, postal = (d.get("address1") or ""), (d.get("city") or ""), (d.get("state") or ""), (d.get("zip") or "")
    if not (a1 and city and postal):
        pa1, pcity, pstate, ppostal = _parse_address(d.get("address"))
        a1 = a1 or pa1; city = city or pcity; state = state or pstate; postal = postal or ppostal
    state = state or SM_DEFAULT_STATE
    if not city and postal in ZIP_CITY:
        city = ZIP_CITY[postal]
    variant = (d.get("version") or "").lower()
    notes = [source_note]
    if d.get("style"): notes.append(f"Fence style: {d['style']}")
    if d.get("lf"):    notes.append(f"Estimated fence length: {d['lf']} ft")
    if d.get("quote"): notes.append(f"Quoted range: {d['quote']}")
    if variant:        notes.append(f"Postcard variant: {variant.upper()}")
    if d.get("page"):  notes.append(f"Landing page: {d['page']}")
    if d.get("test"):  notes.insert(0, "*** TEST LEAD - DO NOT CALL ***")
    tags = "FenceScout" + (f",variant-{variant}" if variant in ("a", "b") else "")
    return {
        "FirstName": first, "LastName": last,
        "Email": d.get("email") or "", "Phone1": d.get("phone") or "", "Phone1Type": "Mobile",
        "Address1": a1, "City": city, "State": state, "PostalCode": postal,
        "Channel": SM_CHANNEL, "Campaign": SM_CAMPAIGN,
        "Tags": tags, "ContactType": "Prospect", "Notes": "\n".join(notes),
    }


def _post_sm(payload):
    """POST a mapped payload to ServiceMinder. Returns (result, contact_id, message)."""
    try:
        r = requests.post(SM_LEAD_URL + SM_LEAD_KEY, data=payload, timeout=20)
        try: j = r.json()
        except Exception: j = {}
        result = j.get("Result", -1 if r.status_code != 200 else 0)
        return result, str(j.get("ContactId", "")), (j.get("Message", "") or (f"HTTP {r.status_code}" if r.status_code != 200 else ""))
    except Exception as e:
        return -1, "", str(e)[:300]


@app.post("/lead")
def lead(req: LeadReq):
    """Landing-form -> ServiceMinder. Fires IMMEDIATELY (warm responders =
    speed-to-lead). The SM key stays server-side."""
    if not SM_LEAD_KEY:
        raise HTTPException(500, "SM_LEAD_KEY not configured")
    p = _map_lead(req.dict())
    result, contact_id, message = _post_sm(p)
    conn = db()
    with conn, conn.cursor() as cur:
        cur.execute("insert into leads(name,email,phone,address1,city,state,postal,variant,page,notes,"
                    "sm_result,sm_contact_id,sm_message) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (req.name, p["Email"], p["Phone1"], p["Address1"], p["City"], p["State"], p["PostalCode"],
                     (req.version or "").lower(), req.page, p["Notes"], result, contact_id, message))
    conn.close()
    if result != 0:
        return {"ok": False, "result": result, "message": message}
    return {"ok": True, "contact_id": contact_id}


@app.get("/leads/recent")
def leads_recent(x_api_key: str = Header(default="")):
    """Quick check that leads are landing (and their SM result). Protected by SCAN_API_KEY if set."""
    _require_key(x_api_key)
    conn = db()
    with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("select created,name,phone,email,city,postal,variant,sm_result,sm_contact_id,sm_message "
                    "from leads order by id desc limit 25")
        rows = cur.fetchall()
    conn.close()
    return {"leads": rows}


# ---------------------------------------------------------------------------
# Outbound queue with delay (postcard recipients who haven't responded yet)
# Holds each lead until ~8 days after queueing so the call happens AFTER the
# postcard lands. DNC numbers are dropped at queue time. Inbound /lead is
# unaffected (still instant).
# ---------------------------------------------------------------------------
OUTBOUND_DELAY_DAYS = int(os.environ.get("OUTBOUND_DELAY_DAYS", "8"))
FLUSH_INTERVAL_SEC = int(os.environ.get("FLUSH_INTERVAL_SEC", "1800"))   # 30 min


class OutboundLead(BaseModel):
    name: str | None = None
    address: str | None = None
    address1: str | None = None
    city: str | None = None
    state: str | None = None
    zip: str | None = None
    phone: str | None = None
    email: str | None = None
    version: str | None = None
    lf: int | None = None
    quote: str | None = None
    style: str | None = None
    dnc: bool | None = False        # True = on Do-Not-Call list -> never queued
    test: bool | None = False


class OutboundReq(BaseModel):
    leads: list[OutboundLead]
    delay_days: int | None = None   # defaults to OUTBOUND_DELAY_DAYS (8)


@app.post("/outbound")
def outbound(req: OutboundReq, x_api_key: str = Header(default="")):
    """Queue postcard recipients for a delayed call. DNC rows are skipped."""
    _require_key(x_api_key)
    delay = OUTBOUND_DELAY_DAYS if req.delay_days is None else req.delay_days
    send_after = datetime.now(timezone.utc) + timedelta(days=delay)
    queued = skipped = 0
    conn = db()
    with conn, conn.cursor() as cur:
        for L in req.leads:
            if L.dnc:
                skipped += 1
                continue
            d = L.dict(); d.pop("dnc", None)
            cur.execute("insert into outbound_queue(send_after,payload,status) values(%s,%s,'queued')",
                        (send_after, json.dumps(d)))
            queued += 1
    conn.close()
    return {"queued": queued, "skipped_dnc": skipped, "delay_days": delay,
            "send_after": send_after.isoformat()}


def flush_due(limit=1000):
    """Send any queued outbound leads whose delay has elapsed."""
    if not SM_LEAD_KEY:
        return {"sent": 0, "errors": 0, "note": "SM_LEAD_KEY not set"}
    conn = db(); sent = errors = 0
    with conn, conn.cursor() as cur:
        cur.execute("select id,payload from outbound_queue where status='queued' "
                    "and send_after<=now() order by id limit %s", (limit,))
        due = cur.fetchall()
    for row_id, payload in due:
        d = json.loads(payload)
        result, cid, msg = _post_sm(_map_lead(d, source_note="FenceScout direct-mail outbound (postcard recipient)."))
        ok = (result == 0)
        sent += ok; errors += (not ok)
        with conn, conn.cursor() as cur:
            cur.execute("update outbound_queue set status=%s, sent_at=now(), sm_result=%s, sm_contact_id=%s, sm_message=%s "
                        "where id=%s", ("sent" if ok else "error", result, cid, msg, row_id))
    conn.close()
    return {"sent": sent, "errors": errors, "due": len(due)}


@app.post("/outbound/flush")
def outbound_flush(x_api_key: str = Header(default="")):
    """Manual trigger (the background thread also runs this every 30 min)."""
    _require_key(x_api_key)
    return flush_due()


@app.get("/outbound/status")
def outbound_status(x_api_key: str = Header(default="")):
    _require_key(x_api_key)
    conn = db()
    with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("select status, count(*) n from outbound_queue group by status")
        counts = {r["status"]: r["n"] for r in cur.fetchall()}
        cur.execute("select min(send_after) next_due from outbound_queue where status='queued'")
        nxt = cur.fetchone()["next_due"]
    conn.close()
    return {"counts": counts, "next_due": nxt.isoformat() if nxt else None,
            "delay_days": OUTBOUND_DELAY_DAYS}


def _flusher_loop():
    while True:
        time.sleep(FLUSH_INTERVAL_SEC)
        try:
            flush_due()
        except Exception:
            pass


# start the background flusher once, in-process (single web worker)
threading.Thread(target=_flusher_loop, daemon=True).start()


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
        lf = int(h.get("lf") or 0) or DEFAULT_LF
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
