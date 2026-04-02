# tests/test_auth_router.py
from unittest.mock import AsyncMock, patch, MagicMock
import pytest


def test_login_redirects_to_google(client):
    resp = client.get("/api/auth/login", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert "accounts.google.com" in resp.headers["location"]


def test_refresh_no_cookie_returns_401(client):
    resp = client.post("/api/auth/refresh")
    assert resp.status_code == 401


def test_refresh_valid_cookie_returns_access_token(client, mocker):
    from app.core.auth import create_refresh_token
    refresh_tok = create_refresh_token({"sub": "uid123"})

    mocker.patch(
        "app.routers.auth.asyncio_to_thread_get_user",
        new_callable=AsyncMock,
        return_value={
            "uid": "uid123", "email": "a@b.com", "name": "A B",
            "avatar": "", "role": "user", "token_balance": 0,
        },
    )
    resp = client.post("/api/auth/refresh", cookies={"refresh_token": refresh_tok})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["user"]["email"] == "a@b.com"


def test_me_requires_auth(client):
    resp = client.get("/api/auth/me")
    assert resp.status_code == 401


def test_me_returns_user_info(client, access_token, mocker):
    mocker.patch(
        "app.routers.auth.asyncio_to_thread_get_user",
        new_callable=AsyncMock,
        return_value={
            "uid": "uid123", "email": "test@example.com", "name": "Test",
            "avatar": "", "role": "user", "token_balance": 10,
        },
    )
    resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {access_token}"})
    assert resp.status_code == 200
    assert resp.json()["email"] == "test@example.com"


def test_logout_clears_cookie(client, access_token):
    resp = client.post(
        "/api/auth/logout",
        headers={"Authorization": f"Bearer {access_token}"},
        cookies={"refresh_token": "some-token"},
    )
    assert resp.status_code == 200
    set_cookie = resp.headers.get("set-cookie", "")
    assert "refresh_token" in set_cookie
    # Cookie is cleared (max-age=0)
    assert "max-age=0" in set_cookie.lower()
