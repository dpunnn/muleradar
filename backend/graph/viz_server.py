"""
viz_server.py — Sajikan graph visualization LIVE dari Neo4j.
Setiap halaman dibuka = query Neo4j saat itu juga (BUKAN file statis).

Jalankan (Neo4j harus nyala via docker compose up -d neo4j):
    cd backend
    uvicorn graph.viz_server:app --port 8050 --reload
Lalu buka: http://localhost:8050

Kontrol interaktif di atas graph:
- Illicit / Clean : ubah jumlah sampel edge -> Query Neo4j ulang
- Seed            : isi account_id -> tampilkan jaringan rekening itu saja
Setiap submit = query baru ke Neo4j -> bukti graph berasal dari dataset, live.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from builder import get_driver
from generate_viz import load_viz_data, build_html  # aman di-import (main sudah diguard)

app = FastAPI(title="MuleRadar — Graph Viz (live dari Neo4j)")

_driver = None
def driver():
    """Driver Neo4j dibuat sekali, dipakai ulang (lazy)."""
    global _driver
    if _driver is None:
        _driver = get_driver()
    return _driver


def load_seed_data(drv, seed: str, limit: int = 400):
    """Ego-network: semua transaksi yang menyentuh akun `seed` (live)."""
    with drv.session() as s:
        rows = s.run("""
            MATCH (a:Account)-[r:TRANSFER]->(b:Account)
            WHERE a.account_id = $seed OR b.account_id = $seed
            RETURN a.account_id AS from_acc, b.account_id AS to_acc,
                   r.amount AS amount, r.channel AS channel,
                   r.is_laundering AS is_laundering
            LIMIT $limit
        """, seed=seed, limit=limit).data()

    illicit = [e for e in rows if e.get("is_laundering") == 1]
    clean   = [e for e in rows if e.get("is_laundering") != 1]
    involved = list({e[k] for e in rows for k in ("from_acc", "to_acc")})

    if not involved:
        return illicit, clean, [], []

    with drv.session() as s:
        deg = s.run("""
            MATCH (a:Account)-[r:TRANSFER]->()
            WHERE a.account_id IN $n
            RETURN a.account_id AS acc, count(r) AS out_deg
        """, n=involved).data()
        ideg = s.run("""
            MATCH (a:Account)-[r:TRANSFER {is_laundering: 1}]->()
            WHERE a.account_id IN $n
            RETURN a.account_id AS acc, count(r) AS illicit_out
        """, n=involved).data()
    return illicit, clean, deg, ideg


def controls(illicit: int, clean: int, seed: str) -> str:
    """Bar kontrol melayang (form GET -> query ulang Neo4j)."""
    inp = ("background:#0d1117;border:1px solid #30363d;color:#e6edf3;"
           "border-radius:6px;padding:4px 8px;font-size:12px;")
    return f"""
  <form method="get" style="position:absolute;top:70px;left:50%;
        transform:translateX(-50%);z-index:10;background:#161b22;
        border:1px solid #30363d;border-radius:8px;padding:8px 12px;
        display:flex;gap:10px;align-items:center;font-size:12px;color:#8b949e;">
    <span style="color:#58a6ff;font-weight:600;">Query Neo4j live:</span>
    <label>Illicit <input name="illicit" type="number" value="{illicit}" style="{inp}width:72px;"></label>
    <label>Clean <input name="clean" type="number" value="{clean}" style="{inp}width:72px;"></label>
    <label>Seed <input name="seed" placeholder="account_id" value="{seed}" style="{inp}width:150px;"></label>
    <button type="submit" style="background:#238636;color:#fff;border:none;
            border-radius:6px;padding:5px 12px;cursor:pointer;font-weight:600;">
      Query Neo4j</button>
  </form>"""


@app.get("/", response_class=HTMLResponse)
def index(
    illicit: int = Query(350),
    clean: int = Query(150),
    seed: str = Query("", max_length=80),
):
    # potong otomatis ke rentang aman (browser tak sanggup render graph raksasa)
    illicit = max(10, min(illicit, 2000))
    clean = max(0, min(clean, 2000))
    drv = driver()
    seed = seed.strip()
    if seed:
        data = load_seed_data(drv, seed)
    else:
        data = load_viz_data(drv, limit_illicit=illicit, limit_clean=clean)

    html = build_html(*data)
    # sisipkan bar kontrol tepat sebelum kanvas graph
    html = html.replace('<div id="mynetwork">',
                        controls(illicit, clean, seed) + '\n  <div id="mynetwork">')
    return HTMLResponse(content=html)


@app.get("/health")
def health():
    try:
        with driver().session() as s:
            n = s.run("MATCH (a:Account) RETURN count(a) AS n").single()["n"]
        return {"status": "ok", "accounts_in_neo4j": n}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.on_event("shutdown")
def _close():
    if _driver is not None:
        _driver.close()
