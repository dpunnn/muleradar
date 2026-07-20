"""
Autentikasi dasar (17-Jul, Prioritas 2 — data AML tanpa proteksi akses adalah
risiko paling berat kalau lupa, lihat PIPELINE.txt bagian audit produksi).

SCOPE SENGAJA MINIMAL, sesuai kebutuhan MVP: satu akun admin dari env var
(bukan tabel users/registrasi — belum ada kebutuhan multi-user/role di scope
ini), JWT bearer token utk semua endpoint API kecuali /auth/login, /health,
/. Kredensial di .env dalam bentuk PLAINTEXT — konsisten dengan konvensi yang
SUDAH ADA di proyek ini (POSTGRES_PASSWORD, NEO4J_PASSWORD juga plaintext di
.env), bukan celah baru.

GANTI AUTH_USERNAME/AUTH_PASSWORD/AUTH_SECRET_KEY di .env sebelum deploy
publik — default di sini cuma fallback dev-lokal.
"""

import os
import time

from fastapi import APIRouter, Depends, Header, HTTPException
from jose import JWTError, jwt
from pydantic import BaseModel

AUTH_USERNAME  = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD  = os.getenv("AUTH_PASSWORD", "muleradar_admin")
SECRET_KEY     = os.getenv("AUTH_SECRET_KEY", "muleradar-dev-secret-CHANGE-IN-PROD")
ALGORITHM      = "HS256"
TOKEN_TTL_S    = int(os.getenv("AUTH_TOKEN_TTL_S", str(8 * 3600)))  # 8 jam (1 shift kerja)

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


def _create_token(username: str) -> str:
    now = int(time.time())
    payload = {"sub": username, "iat": now, "exp": now + TOKEN_TTL_S}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest):
    # Perbandingan langsung (bukan constant-time) — cukup utk MVP single-admin,
    # bukan sistem multi-user yang butuh hardening timing-attack.
    if body.username != AUTH_USERNAME or body.password != AUTH_PASSWORD:
        raise HTTPException(status_code=401, detail="Username atau password salah")
    return LoginResponse(access_token=_create_token(body.username), expires_in=TOKEN_TTL_S)


def require_auth(authorization: str = Header(None)) -> str:
    """Dependency utk semua router selain auth/health — validasi Bearer JWT."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token tidak ada — login dulu")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Token invalid atau kedaluwarsa")
    return payload["sub"]
