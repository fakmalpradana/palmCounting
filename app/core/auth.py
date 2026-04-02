# app/core/auth.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jose import jwt

from app.core.config import settings

ALGORITHM = "HS256"


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    payload = data.copy()
    payload["type"] = "access"
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.jwt_access_expire_minutes)
    )
    payload["exp"] = expire
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def create_refresh_token(data: dict, expires_delta: timedelta | None = None) -> str:
    payload = data.copy()
    payload["type"] = "refresh"
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(days=settings.jwt_refresh_expire_days)
    )
    payload["exp"] = expire
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and verify a JWT. Raises jose.JWTError or jose.ExpiredSignatureError on failure."""
    return jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])


def build_google_oauth_url(state: str) -> str:
    from urllib.parse import urlencode
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": f"{settings.frontend_url}/api/auth/callback",
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "offline",
        "prompt": "select_account",
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
