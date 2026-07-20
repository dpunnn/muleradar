import os
import sys

# Pastikan direktori backend/ ada di sys.path agar sub-package (osint, graph,
# detection) bisa di-import dengan gaya absolut `from osint import ...`,
# konsisten dengan run_detection.py dan consumer.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import auth
from api.routes import osint as osint_routes
from api.routes import dashboard as dashboard_routes
from api.routes import alerts as alerts_routes
from api.routes import graph as graph_routes
from api.routes import cases as cases_routes
from api.routes import copilot as copilot_routes

app = FastAPI(title="MuleRadar API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# /auth/login publik (belum ada token). Semua router data lain WAJIB Bearer
# JWT (fix 17-Jul, Prioritas 2 — data AML tanpa proteksi akses).
app.include_router(auth.router)

_protected = [Depends(auth.require_auth)]
app.include_router(osint_routes.router, dependencies=_protected)
app.include_router(dashboard_routes.router, dependencies=_protected)
app.include_router(alerts_routes.router, dependencies=_protected)
app.include_router(graph_routes.router, dependencies=_protected)
app.include_router(cases_routes.router, dependencies=_protected)
app.include_router(copilot_routes.router, dependencies=_protected)


@app.get("/health")
def health():
    return {"status": "ok", "service": "MuleRadar"}


@app.get("/")
def root():
    return {"message": "MuleRadar API is running"}
