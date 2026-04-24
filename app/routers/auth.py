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
from pydantic import BaseModel, EmailStr, Field

from app.core.auth import (
    build_google_oauth_url,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from app.core.config import settings
from app.core import firestore_client
from app.core.password import hash_password, verify_password
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


# ── Email / Password Auth ────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    name: str = Field(min_length=1)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ResetRequestBody(BaseModel):
    email: EmailStr


class ResetPasswordBody(BaseModel):
    token: str
    new_password: str = Field(min_length=8)


class VerifyEmailBody(BaseModel):
    token: str


class ResendVerificationBody(BaseModel):
    email: EmailStr


@router.post("/register", status_code=201)
async def register(body: RegisterRequest):
    email = body.email.lower()
    existing = await asyncio.to_thread(firestore_client.get_user_by_email, email)

    if existing and existing.get("password_hash"):
        raise HTTPException(409, "Email already registered")

    if existing and not existing.get("password_hash"):
        # Google-only user adding a password — link account.
        await asyncio.to_thread(
            firestore_client.update_user_password,
            existing["uid"],
            hash_password(body.password),
        )
        # Already verified via Google — issue tokens directly.
        access_token = create_access_token({
            "sub": existing["uid"], "email": email, "role": existing["role"],
        })
        refresh_token = create_refresh_token({"sub": existing["uid"]})
        response = JSONResponse(
            status_code=201,
            content={
                "access_token": access_token,
                "token_type": "bearer",
                "message": "Password added to your existing Google account. You are now logged in.",
                "user": _user_response(existing),
            },
        )
        _set_refresh_cookie(response, refresh_token)
        return response

    # New email/password registration.
    user = await asyncio.to_thread(
        firestore_client.create_email_user, email, body.name, hash_password(body.password),
    )

    # Send verification email (import here to avoid circular at module level).
    from app.core.email import send_verification_email
    await asyncio.to_thread(send_verification_email, email, user["uid"])

    return JSONResponse(
        status_code=201,
        content={"message": "Account created. Please check your email to verify your account."},
    )


@router.post("/email-login")
async def email_login(body: LoginRequest):
    email = body.email.lower()
    user = await asyncio.to_thread(firestore_client.get_user_by_email, email)

    if not user or not user.get("password_hash"):
        raise HTTPException(401, "Invalid email or password")

    if not verify_password(body.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")

    if not user.get("email_verified", False):
        raise HTTPException(403, "Please verify your email before logging in")

    access_token = create_access_token({
        "sub": user["uid"], "email": user["email"], "role": user["role"],
    })
    refresh_token = create_refresh_token({"sub": user["uid"]})

    response = JSONResponse({
        "access_token": access_token,
        "token_type": "bearer",
        "user": _user_response(user),
    })
    _set_refresh_cookie(response, refresh_token)
    return response


@router.post("/verify-email")
async def verify_email(body: VerifyEmailBody):
    try:
        payload = decode_token(body.token)
    except JWTError:
        raise HTTPException(401, "Invalid or expired verification link")

    if payload.get("type") != "email_verify":
        raise HTTPException(401, "Invalid token type")

    uid = payload["sub"]
    user = await asyncio.to_thread(firestore_client.get_user, uid)
    if not user:
        raise HTTPException(404, "User not found")

    if not user.get("email_verified", False):
        await asyncio.to_thread(firestore_client.set_email_verified, uid)

    # Refresh user data after verification.
    user = await asyncio.to_thread(firestore_client.get_user, uid)

    access_token = create_access_token({
        "sub": uid, "email": user["email"], "role": user["role"],
    })
    refresh_token = create_refresh_token({"sub": uid})

    response = JSONResponse({
        "access_token": access_token,
        "token_type": "bearer",
        "message": "Email verified successfully",
        "user": _user_response(user),
    })
    _set_refresh_cookie(response, refresh_token)
    return response


@router.post("/request-password-reset")
async def request_password_reset(body: ResetRequestBody):
    user = await asyncio.to_thread(firestore_client.get_user_by_email, body.email.lower())
    if user and user.get("password_hash"):
        from app.core.email import send_password_reset_email
        await asyncio.to_thread(send_password_reset_email, user["email"], user["uid"])

    # Always 200 to prevent email enumeration.
    return {"message": "If that email is registered, a reset link has been sent."}


@router.post("/reset-password")
async def reset_password(body: ResetPasswordBody):
    try:
        payload = decode_token(body.token)
    except JWTError:
        raise HTTPException(401, "Invalid or expired reset link")

    if payload.get("type") != "password_reset":
        raise HTTPException(401, "Invalid token type")

    uid = payload["sub"]
    user = await asyncio.to_thread(firestore_client.get_user, uid)
    if not user:
        raise HTTPException(404, "User not found")

    await asyncio.to_thread(
        firestore_client.update_user_password, uid, hash_password(body.new_password),
    )
    return {"message": "Password updated successfully"}


@router.post("/resend-verification")
async def resend_verification(body: ResendVerificationBody):
    user = await asyncio.to_thread(firestore_client.get_user_by_email, body.email.lower())
    if user and user.get("password_hash") and not user.get("email_verified", False):
        from app.core.email import send_verification_email
        await asyncio.to_thread(send_verification_email, user["email"], user["uid"])

    return {"message": "If that email is registered and unverified, a verification link has been sent."}


def _user_response(user: dict) -> dict:
    """Standard user object returned in auth responses."""
    return {
        "uid": user["uid"],
        "email": user["email"],
        "name": user.get("name", ""),
        "avatar": user.get("avatar", ""),
        "role": user["role"],
        "token_balance": user.get("token_balance", 0),
    }
