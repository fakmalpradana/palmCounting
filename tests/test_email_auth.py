# tests/test_email_auth.py
from unittest.mock import AsyncMock, patch
import pytest

from app.core.auth import create_email_token
from datetime import timedelta


# ── Registration ───────────────────────────────────────────────────────────

def test_register_creates_new_user(client, mocker):
    mocker.patch(
        "app.routers.auth.asyncio.to_thread",
        side_effect=[
            None,  # get_user_by_email → None (no existing user)
            {      # create_email_user → new user
                "uid": "ep_abc", "email": "new@test.com", "name": "New",
                "avatar": "", "role": "user", "token_balance": 0,
                "password_hash": "$2b$...", "auth_provider": "email",
                "email_verified": False,
            },
            None,  # send_verification_email
        ],
    )
    resp = client.post("/api/auth/register", json={
        "email": "new@test.com", "password": "password123", "name": "New",
    })
    assert resp.status_code == 201
    assert "verify" in resp.json()["message"].lower()


def test_register_duplicate_email_returns_409(client, mocker):
    mocker.patch(
        "app.routers.auth.asyncio.to_thread",
        return_value={
            "uid": "ep_existing", "email": "dup@test.com", "name": "Dup",
            "role": "user", "password_hash": "$2b$existing_hash",
        },
    )
    resp = client.post("/api/auth/register", json={
        "email": "dup@test.com", "password": "password123", "name": "Dup",
    })
    assert resp.status_code == 409


def test_register_google_user_adds_password(client, mocker):
    google_user = {
        "uid": "google_123", "email": "guser@test.com", "name": "Google User",
        "avatar": "https://pic.url", "role": "user", "token_balance": 50,
        "password_hash": None, "auth_provider": "google", "email_verified": True,
    }
    calls = []

    async def mock_to_thread(fn, *args):
        calls.append(fn.__name__)
        if fn.__name__ == "get_user_by_email":
            return google_user
        return None  # update_user_password

    mocker.patch("app.routers.auth.asyncio.to_thread", side_effect=mock_to_thread)

    resp = client.post("/api/auth/register", json={
        "email": "guser@test.com", "password": "newpassword", "name": "Google User",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert "access_token" in data
    assert data["user"]["email"] == "guser@test.com"
    assert "update_user_password" in calls


# ── Email Login ────────────────────────────────────────────────────────────

def test_email_login_success(client, mocker):
    from app.core.password import hash_password
    hashed = hash_password("correct-pass")

    async def mock_to_thread(fn, *args):
        return {
            "uid": "ep_user1", "email": "user@test.com", "name": "User",
            "avatar": "", "role": "user", "token_balance": 0,
            "password_hash": hashed, "email_verified": True,
        }

    mocker.patch("app.routers.auth.asyncio.to_thread", side_effect=mock_to_thread)

    resp = client.post("/api/auth/email-login", json={
        "email": "user@test.com", "password": "correct-pass",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["user"]["uid"] == "ep_user1"


def test_email_login_wrong_password(client, mocker):
    from app.core.password import hash_password
    hashed = hash_password("real-password")

    async def mock_to_thread(fn, *args):
        return {
            "uid": "ep_user1", "email": "user@test.com", "name": "User",
            "role": "user", "password_hash": hashed, "email_verified": True,
        }

    mocker.patch("app.routers.auth.asyncio.to_thread", side_effect=mock_to_thread)

    resp = client.post("/api/auth/email-login", json={
        "email": "user@test.com", "password": "wrong-password",
    })
    assert resp.status_code == 401


def test_email_login_unverified_returns_403(client, mocker):
    from app.core.password import hash_password
    hashed = hash_password("mypassword")

    async def mock_to_thread(fn, *args):
        return {
            "uid": "ep_user2", "email": "unverified@test.com", "name": "Unveri",
            "role": "user", "password_hash": hashed, "email_verified": False,
        }

    mocker.patch("app.routers.auth.asyncio.to_thread", side_effect=mock_to_thread)

    resp = client.post("/api/auth/email-login", json={
        "email": "unverified@test.com", "password": "mypassword",
    })
    assert resp.status_code == 403


def test_email_login_unknown_email(client, mocker):
    async def mock_to_thread(fn, *args):
        return None

    mocker.patch("app.routers.auth.asyncio.to_thread", side_effect=mock_to_thread)

    resp = client.post("/api/auth/email-login", json={
        "email": "noone@test.com", "password": "whatever",
    })
    assert resp.status_code == 401


# ── Email Verification ─────────────────────────────────────────────────────

def test_verify_email_valid_token(client, mocker):
    token = create_email_token("ep_user1", "email_verify", timedelta(hours=24))

    call_count = {"n": 0}

    async def mock_to_thread(fn, *args):
        call_count["n"] += 1
        if fn.__name__ == "get_user":
            return {
                "uid": "ep_user1", "email": "user@test.com", "name": "User",
                "avatar": "", "role": "user", "token_balance": 0,
                "email_verified": True,
            }
        return None  # set_email_verified

    mocker.patch("app.routers.auth.asyncio.to_thread", side_effect=mock_to_thread)

    resp = client.post("/api/auth/verify-email", json={"token": token})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["message"] == "Email verified successfully"


def test_verify_email_invalid_token(client):
    resp = client.post("/api/auth/verify-email", json={"token": "bad.token.here"})
    assert resp.status_code == 401


# ── Password Reset ─────────────────────────────────────────────────────────

def test_request_password_reset_always_200(client, mocker):
    async def mock_to_thread(fn, *args):
        return None  # no user found

    mocker.patch("app.routers.auth.asyncio.to_thread", side_effect=mock_to_thread)

    resp = client.post("/api/auth/request-password-reset", json={"email": "nobody@test.com"})
    assert resp.status_code == 200


def test_reset_password_valid_token(client, mocker):
    token = create_email_token("ep_user1", "password_reset", timedelta(hours=1))

    async def mock_to_thread(fn, *args):
        if fn.__name__ == "get_user":
            return {"uid": "ep_user1", "email": "user@test.com", "role": "user"}
        return None  # update_user_password

    mocker.patch("app.routers.auth.asyncio.to_thread", side_effect=mock_to_thread)

    resp = client.post("/api/auth/reset-password", json={
        "token": token, "new_password": "newsecurepass",
    })
    assert resp.status_code == 200
    assert "updated" in resp.json()["message"].lower()


def test_reset_password_invalid_token(client):
    resp = client.post("/api/auth/reset-password", json={
        "token": "invalid.jwt.token", "new_password": "newsecurepass",
    })
    assert resp.status_code == 401
