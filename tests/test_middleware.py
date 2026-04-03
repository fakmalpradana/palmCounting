# tests/test_middleware.py
import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from app.middleware.auth import get_current_user, require_role
from app.core.auth import create_access_token


def _make_app_with_route(dep):
    app = FastAPI()

    @app.get("/protected")
    def protected(user=Depends(dep)):
        return {"uid": user["sub"], "role": user["role"]}

    return app


def test_valid_bearer_token_passes():
    token = create_access_token({"sub": "uid1", "email": "a@b.com", "role": "user"})
    app = _make_app_with_route(get_current_user)
    with TestClient(app) as c:
        resp = c.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["uid"] == "uid1"


def test_missing_token_returns_401():
    app = _make_app_with_route(get_current_user)
    with TestClient(app) as c:
        resp = c.get("/protected")
    assert resp.status_code == 401


def test_malformed_token_returns_401():
    app = _make_app_with_route(get_current_user)
    with TestClient(app) as c:
        resp = c.get("/protected", headers={"Authorization": "Bearer bad-token"})
    assert resp.status_code == 401


def test_require_role_passes_for_correct_role():
    token = create_access_token({"sub": "uid1", "email": "a@b.com", "role": "admin"})
    app = _make_app_with_route(require_role("admin", "superadmin"))
    with TestClient(app) as c:
        resp = c.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200


def test_require_role_returns_403_for_wrong_role():
    token = create_access_token({"sub": "uid1", "email": "a@b.com", "role": "user"})
    app = _make_app_with_route(require_role("admin", "superadmin"))
    with TestClient(app) as c:
        resp = c.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403
