# app/routers/auth.py
from __future__ import annotations

import asyncio
import secrets

import logging
import traceback

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from google.api_core import exceptions as gcp_exceptions
from jose import JWTError
from jose import jwt as jose_jwt

from app.core.auth import (
    build_google_oauth_url,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from app.core.config import settings
from app.core import firestore_client
from app.middleware.auth import get_current_user

router = APIRouter(prefix="/api/auth", tags=["auth"])

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


def _is_secure() -> bool:
    return settings.frontend_url.startswith("https://")


def _set_refresh_cookie(response, token: str) -> None:
    secure = _is_secure()
    response.set_cookie(
        key="refresh_token",
        value=token,
        httponly=True,
        secure=secure,
        samesite="strict" if secure else "lax",
        max_age=settings.jwt_refresh_expire_days * 86400,
        path="/",
    )


async def asyncio_to_thread_get_user(uid: str):
    return await asyncio.to_thread(firestore_client.get_user, uid)


async def asyncio_to_thread_upsert_user(uid: str, email: str, name: str, avatar: str):
    return await asyncio.to_thread(firestore_client.upsert_user, uid, email, name, avatar)


@router.get("/dev-login")
async def dev_login():
    """
    Local development only — bypasses Google OAuth and Firestore entirely.
    Only active when DEV_MODE=true in .env.  Returns 404 in production.
    """
    if not settings.dev_mode:
        raise HTTPException(404, "Not found")

    from app.core import firestore_client as _fc
    dev_user = _fc._DEV_USER

    access_token  = create_access_token({
        "sub":   dev_user["uid"],
        "email": dev_user["email"],
        "role":  dev_user["role"],
    })
    refresh_token = create_refresh_token({"sub": dev_user["uid"]})

    response = RedirectResponse(settings.frontend_url)
    _set_refresh_cookie(response, refresh_token)
    return response


@router.get("/login")
async def login():
    state = secrets.token_urlsafe(16)
    url = build_google_oauth_url(state)
    response = RedirectResponse(url)
    secure = _is_secure()
    response.set_cookie(
        "oauth_state", state,
        max_age=600, httponly=True,
        secure=secure, samesite="lax",
    )
    return response


@router.get("/callback")
async def callback(code: str, state: str, request: Request):
    stored_state = request.cookies.get("oauth_state")
    if not stored_state or stored_state != state:
        raise HTTPException(400, "Invalid OAuth state — possible CSRF attempt")

    try:
        async with httpx.AsyncClient() as http:
            token_resp = await http.post(
                GOOGLE_TOKEN_URL,
                data={
                    "code": code,
                    "client_id": settings.google_client_id,
                    "client_secret": settings.google_client_secret,
                    "redirect_uri": f"{settings.frontend_url}/api/auth/callback",
                    "grant_type": "authorization_code",
                },
            )

        if token_resp.status_code != 200:
            raise HTTPException(400, f"Google token exchange failed: {token_resp.text}")

        token_data = token_resp.json()
        id_token_raw = token_data.get("id_token")
        if not id_token_raw:
            raise HTTPException(400, "No id_token in Google response")

        # Decode without signature verification — we trust it; we just exchanged the code
        # directly from Google's token endpoint. Also skip at_hash validation since
        # python-jose requires passing the raw access_token for that check, which is
        # unnecessary given we already validated the exchange server-side.
        google_payload = jose_jwt.decode(
            id_token_raw,
            key="",
            options={
                "verify_signature": False,
                "verify_aud": False,
                "verify_at_hash": False,
            },
        )

        google_uid = google_payload["sub"]
        email = google_payload.get("email", "")
        name = google_payload.get("name", email)
        avatar = google_payload.get("picture", "")

        try:
            user = await asyncio_to_thread_upsert_user(google_uid, email, name, avatar)
        except gcp_exceptions.GoogleAPICallError as exc:
            logging.error("Firestore upsert_user failed: %s\n%s", exc, traceback.format_exc())
            raise HTTPException(500, f"Database error during login: {type(exc).__name__}: {exc}")

        access_token = create_access_token({"sub": google_uid, "email": email, "role": user["role"]})
        refresh_token = create_refresh_token({"sub": google_uid})

        response = RedirectResponse(settings.frontend_url)
        _set_refresh_cookie(response, refresh_token)
        response.delete_cookie("oauth_state")
        return response

    except HTTPException:
        raise
    except Exception as exc:
        logging.error("OAuth callback error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, f"OAuth callback failed: {type(exc).__name__}: {exc}")


@router.post("/refresh")
async def refresh(request: Request):
    raw = request.cookies.get("refresh_token")
    if not raw:
        raise HTTPException(401, "No refresh token cookie")

    try:
        payload = decode_token(raw)
    except JWTError:
        raise HTTPException(401, "Refresh token invalid or expired")

    if payload.get("type") != "refresh":
        raise HTTPException(401, "Invalid token type")

    user = await asyncio_to_thread_get_user(payload["sub"])
    if not user:
        raise HTTPException(401, "User not found")

    access_token = create_access_token({
        "sub": user["uid"],
        "email": user["email"],
        "role": user["role"],
    })

    return JSONResponse({
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "uid": user["uid"],
            "email": user["email"],
            "name": user["name"],
            "avatar": user.get("avatar", ""),
            "role": user["role"],
            "token_balance": user.get("token_balance", 0),
        },
    })


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    user = await asyncio_to_thread_get_user(current_user["sub"])
    if not user:
        raise HTTPException(404, "User not found")
    return {
        "uid": user["uid"],
        "email": user["email"],
        "name": user["name"],
        "avatar": user.get("avatar", ""),
        "role": user["role"],
        "token_balance": user.get("token_balance", 0),
    }


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    response = JSONResponse({"message": "Logged out"})
    secure = _is_secure()
    response.set_cookie(
        "refresh_token", "",
        max_age=0,
        path="/",
        httponly=True,
        secure=secure,
        samesite="strict" if secure else "lax",
    )
    return response
