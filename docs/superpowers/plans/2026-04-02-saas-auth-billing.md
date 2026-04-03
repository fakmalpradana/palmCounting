# SaaS Auth, Billing & GPU Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform Palm Counter into a SaaS platform with Google OAuth login, Firestore user management, role-based access control, free/commercial tier billing, and GPU Cloud Run infrastructure.

**Architecture:** FastAPI backend gains an auth layer (Google OAuth → JWT in memory + httpOnly refresh cookie), a Firestore user store, and a billing engine that gates inference on tier/token logic. The existing CPU Cloud Run service handles free-tier requests; a new GPU Cloud Run service handles commercial requests in Step 4.

**Tech Stack:** `python-jose[cryptography]` (JWT), `httpx` (OAuth exchange), `google-cloud-firestore` (user store), `pydantic-settings` (config), `google-cloud-storage` (signed URLs, Step 4), `pytest` + `pytest-asyncio` + `pytest-mock` (testing)

---

## ⚠️ PAUSE CHECKPOINTS

This plan has 4 mandatory pause points — one per step. After each step's final commit, **stop and wait for user confirmation** before proceeding to the next step.

---

## File Map

### New Files
| File | Responsibility |
|---|---|
| `app/core/config.py` | Pydantic settings, reads from `.env` |
| `app/core/auth.py` | JWT create/decode, Google OAuth helpers |
| `app/core/firestore_client.py` | Firestore CRUD for users (lazy init, sync) |
| `app/middleware/auth.py` | `get_current_user` + `require_role` FastAPI deps |
| `app/routers/auth.py` | `/api/auth/*` endpoints |
| `app/routers/admin.py` | `/api/admin/*` endpoints |
| `tests/__init__.py` | Package marker |
| `tests/conftest.py` | Fixtures: env vars, mocked Firestore, TestClient |
| `tests/test_auth_utils.py` | JWT encode/decode unit tests |
| `tests/test_auth_router.py` | Auth endpoint tests |
| `tests/test_admin_router.py` | Admin endpoint tests |
| `tests/test_billing.py` | Token calculation + tier logic tests |
| `infra/deploy-cpu.sh` | Update CPU Cloud Run service config |
| `infra/deploy-gpu.sh` | Deploy GPU Cloud Run service |
| `infra/setup-gpu-sa.sh` | Grant service account GCS + Firestore access |

### Modified Files
| File | Change |
|---|---|
| `requirements.txt` | Add new deps |
| `app/main.py` | Add auth + admin routers, update CORS |
| `app/routers/inference.py` | Add auth dependency, tier routing, token deduction |
| `static/index.html` | Login screen, silent refresh, auth headers, admin panel |
| `.github/workflows/deploy.yml` | Pass new env vars to Cloud Run |

---

## ════════════════════════════════════════════
## STEP 1: Database & Google OAuth Integration
## ════════════════════════════════════════════

### Task 1: Add dependencies + config

**Files:**
- Modify: `requirements.txt`
- Create: `app/core/config.py`

- [ ] **Step 1: Update requirements.txt**

Append to `requirements.txt`:
```
# Auth & config
python-jose[cryptography]>=3.3.0
httpx>=0.27.0
pydantic-settings>=2.0.0
google-cloud-firestore>=2.16.0
google-cloud-storage>=2.16.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-mock>=3.12.0
```

- [ ] **Step 2: Create app/core/config.py**

```python
# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    google_client_id: str = ""
    google_client_secret: str = ""
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_access_expire_minutes: int = 15
    jwt_refresh_expire_days: int = 7
    firestore_project_id: str = ""
    frontend_url: str = "http://localhost:8080"
    gpu_worker_url: str = ""
    gcs_bucket_name: str = ""
    cleanup_max_age_hours: int = 24
    cleanup_secret: str = ""


settings = Settings()
```

- [ ] **Step 3: Install dependencies**

```bash
pip install python-jose[cryptography] httpx pydantic-settings google-cloud-firestore google-cloud-storage pytest pytest-asyncio pytest-mock
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt app/core/config.py
git commit -m "chore(deps): add auth, firestore, and testing dependencies"
```

---

### Task 2: Test infrastructure + Firestore client

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `app/core/firestore_client.py`
- Create: `tests/test_firestore_client.py`

- [ ] **Step 1: Create tests/__init__.py**

```python
# tests/__init__.py
```

- [ ] **Step 2: Create tests/conftest.py**

```python
# tests/conftest.py
import os

# Set env vars BEFORE any app imports
os.environ.setdefault("GOOGLE_CLIENT_ID", "test-client-id")
os.environ.setdefault("GOOGLE_CLIENT_SECRET", "test-client-secret")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-32-bytes-long-ok!")
os.environ.setdefault("FIRESTORE_PROJECT_ID", "test-project")
os.environ.setdefault("FRONTEND_URL", "http://localhost:8080")

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def mock_firestore_db(mocker):
    """Patch the Firestore lazy getter so no real DB connection is made."""
    mock_db = MagicMock()
    mocker.patch("app.core.firestore_client.get_db", return_value=mock_db)
    return mock_db


@pytest.fixture
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def access_token():
    from app.core.auth import create_access_token
    return create_access_token({"sub": "uid123", "email": "test@example.com", "role": "user"})


@pytest.fixture
def admin_token():
    from app.core.auth import create_access_token
    return create_access_token({"sub": "uid_admin", "email": "admin@example.com", "role": "admin"})


@pytest.fixture
def superadmin_token():
    from app.core.auth import create_access_token
    return create_access_token({"sub": "uid_super", "email": "fakmalpradana@gmail.com", "role": "superadmin"})
```

- [ ] **Step 3: Create app/core/firestore_client.py**

```python
# app/core/firestore_client.py
from __future__ import annotations

from datetime import date
from typing import Optional

from app.core.config import settings

SUPERADMIN_EMAIL = "fakmalpradana@gmail.com"

_db = None


def get_db():
    global _db
    if _db is None:
        from google.cloud import firestore
        _db = firestore.Client(project=settings.firestore_project_id)
    return _db


def upsert_user(google_uid: str, email: str, name: str, avatar: str) -> dict:
    db = get_db()
    ref = db.collection("users").document(google_uid)
    doc = ref.get()

    if email == SUPERADMIN_EMAIL:
        role = "superadmin"
    elif doc.exists:
        role = doc.to_dict().get("role", "user")
    else:
        role = "user"

    if doc.exists:
        existing = doc.to_dict()
        updates = {"email": email, "name": name, "avatar": avatar, "role": role}
        ref.update(updates)
        return {**existing, **updates, "uid": google_uid}
    else:
        data = {
            "email": email,
            "name": name,
            "avatar": avatar,
            "role": role,
            "token_balance": 0,
            "daily_upload_count": 0,
            "last_upload_date": "",
        }
        ref.set(data)
        return {**data, "uid": google_uid}


def get_user(google_uid: str) -> Optional[dict]:
    db = get_db()
    doc = db.collection("users").document(google_uid).get()
    if not doc.exists:
        return None
    return {**doc.to_dict(), "uid": google_uid}


def get_all_users() -> list[dict]:
    db = get_db()
    return [{**doc.to_dict(), "uid": doc.id} for doc in db.collection("users").stream()]


def update_token_balance(google_uid: str, delta: int) -> int:
    """Add or deduct tokens (delta can be negative). Returns new balance. Never goes below 0."""
    from google.cloud import firestore

    db = get_db()
    ref = db.collection("users").document(google_uid)

    @firestore.transactional
    def _txn(transaction, ref):
        snap = ref.get(transaction=transaction)
        if not snap.exists:
            raise ValueError("User not found")
        current = snap.to_dict().get("token_balance", 0)
        new_balance = max(0, current + delta)
        transaction.update(ref, {"token_balance": new_balance})
        return new_balance

    return _txn(db.transaction(), ref)


def deduct_tokens(google_uid: str, amount: int) -> int:
    """Atomically deduct tokens. Raises ValueError if balance insufficient."""
    from google.cloud import firestore

    db = get_db()
    ref = db.collection("users").document(google_uid)

    @firestore.transactional
    def _txn(transaction, ref):
        snap = ref.get(transaction=transaction)
        if not snap.exists:
            raise ValueError("User not found")
        current = snap.to_dict().get("token_balance", 0)
        if current < amount:
            raise ValueError(f"Insufficient tokens: have {current}, need {amount}")
        new_balance = current - amount
        transaction.update(ref, {"token_balance": new_balance})
        return new_balance

    return _txn(db.transaction(), ref)


def update_role(google_uid: str, new_role: str) -> None:
    """Change a user's role. Raises ValueError for superadmin or unknown user."""
    db = get_db()
    ref = db.collection("users").document(google_uid)
    doc = ref.get()
    if not doc.exists:
        raise ValueError("User not found")
    if doc.to_dict().get("email") == SUPERADMIN_EMAIL:
        raise ValueError("Cannot change superadmin role")
    ref.update({"role": new_role})


def check_and_increment_daily_upload(google_uid: str) -> dict:
    """Reset counter if date changed, increment, return updated counts.

    Raises ValueError if daily limit (3) is reached.
    """
    from google.cloud import firestore

    db = get_db()
    ref = db.collection("users").document(google_uid)
    today = date.today().isoformat()

    @firestore.transactional
    def _txn(transaction, ref):
        snap = ref.get(transaction=transaction)
        data = snap.to_dict()
        last_date = data.get("last_upload_date", "")
        count = data.get("daily_upload_count", 0) if last_date == today else 0
        if count >= 3:
            raise ValueError("Daily upload limit reached")
        transaction.update(ref, {
            "daily_upload_count": count + 1,
            "last_upload_date": today,
        })
        return {"daily_upload_count": count + 1, "last_upload_date": today}

    return _txn(db.transaction(), ref)
```

- [ ] **Step 4: Write tests for firestore_client**

Create `tests/test_firestore_client.py`:

```python
# tests/test_firestore_client.py
from unittest.mock import MagicMock, patch
import pytest


def _make_doc(exists: bool, data: dict = None):
    doc = MagicMock()
    doc.exists = exists
    doc.to_dict.return_value = data or {}
    return doc


def test_upsert_user_new(mocker):
    mock_db = MagicMock()
    ref = MagicMock()
    mock_db.collection.return_value.document.return_value = ref
    ref.get.return_value = _make_doc(False)
    mocker.patch("app.core.firestore_client.get_db", return_value=mock_db)

    from app.core.firestore_client import upsert_user
    result = upsert_user("uid1", "test@example.com", "Test User", "http://avatar.url")

    ref.set.assert_called_once()
    assert result["role"] == "user"
    assert result["token_balance"] == 0
    assert result["uid"] == "uid1"


def test_upsert_user_superadmin_always_gets_superadmin_role(mocker):
    mock_db = MagicMock()
    ref = MagicMock()
    mock_db.collection.return_value.document.return_value = ref
    ref.get.return_value = _make_doc(True, {"role": "user", "token_balance": 0,
                                             "daily_upload_count": 0, "last_upload_date": ""})
    mocker.patch("app.core.firestore_client.get_db", return_value=mock_db)

    from app.core.firestore_client import upsert_user
    result = upsert_user("uid_super", "fakmalpradana@gmail.com", "Akmal", "")

    assert result["role"] == "superadmin"


def test_upsert_user_existing_preserves_role(mocker):
    mock_db = MagicMock()
    ref = MagicMock()
    mock_db.collection.return_value.document.return_value = ref
    ref.get.return_value = _make_doc(True, {"role": "admin", "token_balance": 50,
                                             "daily_upload_count": 0, "last_upload_date": ""})
    mocker.patch("app.core.firestore_client.get_db", return_value=mock_db)

    from app.core.firestore_client import upsert_user
    result = upsert_user("uid_admin", "other@example.com", "Other", "")

    assert result["role"] == "admin"
    assert result["token_balance"] == 50


def test_get_user_not_found(mocker):
    mock_db = MagicMock()
    mock_db.collection.return_value.document.return_value.get.return_value = _make_doc(False)
    mocker.patch("app.core.firestore_client.get_db", return_value=mock_db)

    from app.core.firestore_client import get_user
    assert get_user("nonexistent") is None


def test_update_role_raises_for_superadmin(mocker):
    mock_db = MagicMock()
    ref = MagicMock()
    mock_db.collection.return_value.document.return_value = ref
    ref.get.return_value = _make_doc(True, {"email": "fakmalpradana@gmail.com"})
    mocker.patch("app.core.firestore_client.get_db", return_value=mock_db)

    from app.core.firestore_client import update_role
    with pytest.raises(ValueError, match="Cannot change superadmin role"):
        update_role("uid_super", "user")
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_firestore_client.py -v
```

Expected: 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add tests/__init__.py tests/conftest.py app/core/firestore_client.py tests/test_firestore_client.py
git commit -m "feat(db): add Firestore client with user CRUD and superadmin enforcement"
```

---

### Task 3: JWT utilities

**Files:**
- Create: `app/core/auth.py`
- Create: `tests/test_auth_utils.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_auth_utils.py`:

```python
# tests/test_auth_utils.py
from datetime import timedelta
import pytest


def test_access_token_contains_correct_claims():
    from app.core.auth import create_access_token, decode_token
    token = create_access_token({"sub": "uid1", "email": "a@b.com", "role": "user"})
    payload = decode_token(token)
    assert payload["sub"] == "uid1"
    assert payload["email"] == "a@b.com"
    assert payload["role"] == "user"
    assert payload["type"] == "access"


def test_refresh_token_contains_correct_claims():
    from app.core.auth import create_refresh_token, decode_token
    token = create_refresh_token({"sub": "uid1"})
    payload = decode_token(token)
    assert payload["sub"] == "uid1"
    assert payload["type"] == "refresh"


def test_decode_expired_token_raises():
    from app.core.auth import create_access_token, decode_token
    from jose import ExpiredSignatureError
    token = create_access_token({"sub": "uid1"}, expires_delta=timedelta(seconds=-1))
    with pytest.raises(ExpiredSignatureError):
        decode_token(token)


def test_decode_invalid_token_raises():
    from app.core.auth import decode_token
    from jose import JWTError
    with pytest.raises(JWTError):
        decode_token("not.a.valid.token")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_auth_utils.py -v
```

Expected: ImportError or 4 FAIL (module doesn't exist yet)

- [ ] **Step 3: Create app/core/auth.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_auth_utils.py -v
```

Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/auth.py tests/test_auth_utils.py
git commit -m "feat(auth): add JWT create/decode utilities"
```

---

### Task 4: Auth middleware

**Files:**
- Create: `app/middleware/auth.py`
- Create: `tests/test_middleware.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_middleware.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_middleware.py -v
```

Expected: ImportError or 5 FAIL

- [ ] **Step 3: Create app/middleware/auth.py**

```python
# app/middleware/auth.py
from fastapi import Depends, HTTPException, Request
from jose import JWTError

from app.core.auth import decode_token


async def get_current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth[len("Bearer "):]
    try:
        payload = decode_token(token)
    except JWTError:
        raise HTTPException(status_code=401, detail="Token is invalid or expired")
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")
    return payload


def require_role(*roles: str):
    """Dependency factory — use as Depends(require_role('admin', 'superadmin'))."""
    async def _check(user: dict = Depends(get_current_user)):
        if user.get("role") not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return _check
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_middleware.py -v
```

Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add app/middleware/__init__.py app/middleware/auth.py tests/test_middleware.py
git commit -m "feat(auth): add JWT middleware with get_current_user and require_role dependencies"
```

Note: create `app/middleware/__init__.py` as an empty file.

---

### Task 5: Auth router

**Files:**
- Create: `app/routers/auth.py`
- Create: `tests/test_auth_router.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_auth_router.py`:

```python
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
        return_value={
            "uid": "uid123", "email": "test@example.com", "name": "Test",
            "avatar": "", "role": "user", "token_balance": 10,
        },
    )
    resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {access_token}"})
    assert resp.status_code == 200
    assert resp.json()["email"] == "test@example.com"


def test_logout_clears_cookie(client, access_token):
    resp = client.post("/api/auth/logout",
                       headers={"Authorization": f"Bearer {access_token}"},
                       cookies={"refresh_token": "some-token"})
    assert resp.status_code == 200
    # Verify cookie is cleared (max-age=0 or expires in past)
    set_cookie = resp.headers.get("set-cookie", "")
    assert "refresh_token" in set_cookie
    assert "max-age=0" in set_cookie.lower() or "expires=" in set_cookie.lower()
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_auth_router.py -v
```

Expected: ImportError or all FAIL

- [ ] **Step 3: Create app/routers/auth.py**

```python
# app/routers/auth.py
from __future__ import annotations

import asyncio
import secrets

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from jose import JWTError, jwt as jose_jwt

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

    id_token_raw = token_resp.json().get("id_token")
    if not id_token_raw:
        raise HTTPException(400, "No id_token in Google response")

    # Decode without signature verification — we trust it; we just exchanged the code
    google_payload = jose_jwt.decode(
        id_token_raw,
        key="",
        options={"verify_signature": False, "verify_aud": False},
    )

    google_uid = google_payload["sub"]
    email = google_payload.get("email", "")
    name = google_payload.get("name", email)
    avatar = google_payload.get("picture", "")

    user = await asyncio_to_thread_upsert_user(google_uid, email, name, avatar)

    access_token = create_access_token({"sub": google_uid, "email": email, "role": user["role"]})
    refresh_token = create_refresh_token({"sub": google_uid})

    response = RedirectResponse(settings.frontend_url)
    _set_refresh_cookie(response, refresh_token)
    response.delete_cookie("oauth_state")
    return response


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
    response.set_cookie("refresh_token", "", max_age=0, path="/")
    return response
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_auth_router.py -v
```

Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add app/routers/auth.py tests/test_auth_router.py
git commit -m "feat(auth): add Google OAuth callback, refresh, me, and logout endpoints"
```

---

### Task 6: Wire into main.py + protect inference routes

**Files:**
- Modify: `app/main.py`
- Modify: `app/routers/inference.py`

- [ ] **Step 1: Update app/main.py**

Replace the entire file:

```python
# app/main.py
# Copyright © 2026 Geo AI Twinverse.
# Contributors: Fikri Kurniawan, Fairuz Akmal Pradana

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.routers import inference as inference_router
from app.routers import auth as auth_router

app = FastAPI(
    title="Palm Counter API",
    description="YOLO ONNX palm-tree detection — GeoTIFF in, GeoJSON out.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(inference_router.router)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


# Serve the frontend from ./static at root
app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

- [ ] **Step 2: Add auth to inference routes**

In `app/routers/inference.py`, add these imports near the top (after existing imports):

```python
from fastapi import Depends
from app.middleware.auth import get_current_user
```

Then update the four protected endpoint signatures:

```python
# POST /api/inference
@router.post("/inference")
async def infer(
    file: UploadFile = File(...),
    model_name: str = Form("best_1.onnx"),
    tile_width: int = Form(640),
    tile_height: int = Form(640),
    min_distance: float = Form(1.0),
    conf_threshold: float = Form(0.25),
    nms_threshold: float = Form(0.4),
    current_user: dict = Depends(get_current_user),   # ← ADD
):
```

```python
# GET /api/download/{file_id}
@router.get("/download/{file_id}")
def download_result(file_id: str, current_user: dict = Depends(get_current_user)):   # ← ADD
```

```python
# GET /api/preview/{file_id}
@router.get("/preview/{file_id}")
def get_preview(file_id: str, current_user: dict = Depends(get_current_user)):   # ← ADD
```

```python
# GET /api/models
@router.get("/models")
def list_models(current_user: dict = Depends(get_current_user)):   # ← ADD
```

```python
# POST /api/models
@router.post("/models")
async def upload_model(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):   # ← ADD
```

`/api/cleanup` remains unauthenticated (uses CLEANUP_SECRET).

- [ ] **Step 3: Run all tests**

```bash
pytest tests/ -v
```

Expected: All existing tests PASS. Auth headers are now required for inference routes.

- [ ] **Step 4: Quick smoke test — verify 401 on protected route**

```bash
curl -s http://localhost:8080/api/models | python3 -m json.tool
```

Expected: `{"detail": "Missing or invalid Authorization header"}`

(Run `uvicorn app.main:app --port 8080` in another terminal first)

- [ ] **Step 5: Commit**

```bash
git add app/main.py app/routers/inference.py app/middleware/__init__.py
git commit -m "feat(auth): wire auth router into app and protect inference endpoints"
```

---

### Task 7: Frontend auth UI

**Files:**
- Modify: `static/index.html`

- [ ] **Step 1: Add auth state variable and login screen**

In `static/index.html`, find the `<script>` tag (near bottom of file). Add these variables at the very top of the script, before any other JS:

```javascript
// ── Auth state (in memory only — never persisted) ──────────────────────
let accessToken = null;
let currentUser = null;
```

- [ ] **Step 2: Add the login screen HTML**

Just before the existing first `<div>` screen (the upload screen), add:

```html
<!-- ── Login Screen ─────────────────────────────────── -->
<div id="screen-login" class="screen active">
  <div class="login-container">
    <h1 class="login-title">PALM COUNTER</h1>
    <p class="login-subtitle">WebGIS YOLO Detection Platform</p>
    <a href="/api/auth/login" class="btn-google-login">
      <svg width="18" height="18" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg">
        <path d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.875 2.684-6.615z" fill="#4285F4"/>
        <path d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z" fill="#34A853"/>
        <path d="M3.964 10.71A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.042l3.007-2.332z" fill="#FBBC05"/>
        <path d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.958L3.964 6.29C4.672 4.163 6.656 3.58 9 3.58z" fill="#EA4335"/>
      </svg>
      Sign in with Google
    </a>
  </div>
</div>
```

- [ ] **Step 3: Add login screen CSS**

In the `<style>` section, add:

```css
.login-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
  gap: 1.5rem;
  background: #0a0a0a;
}
.login-title {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 3.5rem;
  letter-spacing: 0.15em;
  color: #fff;
  margin: 0;
}
.login-subtitle {
  color: #888;
  font-size: 0.9rem;
  margin: 0;
}
.btn-google-login {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  background: #fff;
  color: #333;
  padding: 0.75rem 1.5rem;
  border-radius: 4px;
  text-decoration: none;
  font-size: 0.95rem;
  font-weight: 500;
  transition: box-shadow 0.2s;
}
.btn-google-login:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
```

- [ ] **Step 4: Add auth initialization logic**

Add a `showScreen` function and `initAuth` function in the script (replace or extend any existing screen-switching logic):

```javascript
// ── Screen management ──────────────────────────────────────────────────
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

// ── Auth initialization (runs on every page load) ──────────────────────
async function initAuth() {
  try {
    const resp = await fetch('/api/auth/refresh', {
      method: 'POST',
      credentials: 'include',   // sends httpOnly cookie automatically
    });
    if (!resp.ok) throw new Error('No session');
    const data = await resp.json();
    accessToken = data.access_token;
    currentUser = data.user;
    onAuthSuccess();
  } catch {
    showScreen('screen-login');
  }
}

function onAuthSuccess() {
  updateUserHeader();
  showScreen('screen-upload');
}

function updateUserHeader() {
  const balanceEl = document.getElementById('token-balance');
  const nameEl = document.getElementById('user-name');
  const adminLink = document.getElementById('admin-link');
  if (balanceEl) balanceEl.textContent = currentUser.token_balance.toLocaleString();
  if (nameEl) nameEl.textContent = currentUser.name;
  if (adminLink) {
    adminLink.style.display = ['admin', 'superadmin'].includes(currentUser.role) ? 'inline' : 'none';
  }
}

// ── Authenticated fetch wrapper (auto-retries once on 401) ─────────────
async function authFetch(url, options = {}) {
  options.headers = { ...(options.headers || {}), 'Authorization': `Bearer ${accessToken}` };
  let resp = await fetch(url, options);
  if (resp.status === 401) {
    // Try silent refresh once
    try {
      const r = await fetch('/api/auth/refresh', { method: 'POST', credentials: 'include' });
      if (!r.ok) throw new Error();
      const d = await r.json();
      accessToken = d.access_token;
      currentUser = d.user;
      options.headers['Authorization'] = `Bearer ${accessToken}`;
      resp = await fetch(url, options);
    } catch {
      showScreen('screen-login');
      throw new Error('Session expired');
    }
  }
  return resp;
}
```

- [ ] **Step 5: Replace all bare `fetch(` calls with `authFetch(`**

In `static/index.html`, find every `fetch('/api/` call (except `/api/auth/refresh`) and replace `fetch(` with `authFetch(`. There should be ~5 calls:
- `fetch('/api/models'` → `authFetch('/api/models'`
- `fetch('/api/inference'` → `authFetch('/api/inference'`
- `fetch('/api/download/` → `authFetch('/api/download/`
- `fetch('/api/preview/` → `authFetch('/api/preview/`
- `fetch('/api/models', { method: 'POST'` → `authFetch('/api/models', { method: 'POST'`

- [ ] **Step 6: Add user header bar HTML**

Inside the upload screen's top area (or as a fixed header), add:

```html
<div class="user-header" id="user-header">
  <span id="user-name"></span>
  <span class="token-info">
    Tokens: <strong id="token-balance">0</strong>
    <span class="token-rate">(1 Token = IDR 1.000)</span>
  </span>
  <a href="#" id="admin-link" style="display:none" onclick="showAdminPanel()">Admin</a>
  <a href="#" onclick="logout()">Logout</a>
</div>
```

Add CSS:
```css
.user-header {
  position: fixed; top: 0; right: 0;
  display: flex; align-items: center; gap: 1rem;
  padding: 0.5rem 1.5rem;
  background: rgba(0,0,0,0.7); color: #ccc;
  font-size: 0.8rem; z-index: 1000;
}
.token-rate { color: #666; font-size: 0.75rem; }
```

- [ ] **Step 7: Add logout function and boot call**

```javascript
async function logout() {
  await fetch('/api/auth/logout', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${accessToken}` },
    credentials: 'include',
  });
  accessToken = null;
  currentUser = null;
  showScreen('screen-login');
}

// Boot
document.addEventListener('DOMContentLoaded', initAuth);
```

- [ ] **Step 8: Test locally**

```bash
uvicorn app.main:app --port 8080 --reload
```

Open `http://localhost:8080` — should show login screen. Click "Sign in with Google" → completes OAuth → redirected back → shows upload screen.

- [ ] **Step 9: Commit — STEP 1 COMPLETE**

```bash
git add static/index.html
git commit -m "feat(auth): add login screen, silent refresh, and auth header injection to frontend"
```

Then make the Step 1 final commit:

```bash
git commit --allow-empty -m "feat(auth): integrate Google OAuth and user database

- Firestore user store with superadmin enforcement
- JWT access token (memory) + refresh token (httpOnly cookie)
- Google OAuth 2.0 Authorization Code Flow
- Auth middleware protecting all inference endpoints
- Login screen with silent refresh on page load

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

> ## ⏸️ PAUSE CHECKPOINT 1
> Step 1 complete. Stop here and wait for user confirmation before proceeding to Step 2.

---

## ══════════════════════════════════════
## STEP 2: Role Management & Admin Dashboard
## ══════════════════════════════════════

### Task 8: Admin router

**Files:**
- Create: `app/routers/admin.py`
- Create: `tests/test_admin_router.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_admin_router.py`:

```python
# tests/test_admin_router.py
from unittest.mock import MagicMock


def _user(uid, email, role, balance):
    return {"uid": uid, "email": email, "name": "Test", "avatar": "",
            "role": role, "token_balance": balance,
            "daily_upload_count": 0, "last_upload_date": ""}


def test_list_users_requires_admin(client, access_token):
    resp = client.get("/api/admin/users", headers={"Authorization": f"Bearer {access_token}"})
    assert resp.status_code == 403


def test_list_users_returns_users(client, admin_token, mocker):
    mocker.patch("app.routers.admin.asyncio_to_thread_get_all_users",
                 return_value=[_user("uid1", "a@b.com", "user", 0)])
    resp = client.get("/api/admin/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    assert len(resp.json()["users"]) == 1


def test_update_tokens_adds_correctly(client, admin_token, mocker):
    mocker.patch("app.routers.admin.asyncio_to_thread_update_tokens", return_value=150)
    resp = client.patch("/api/admin/users/uid1/tokens",
                        json={"delta": 150},
                        headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    assert resp.json()["new_balance"] == 150


def test_update_tokens_requires_admin(client, access_token):
    resp = client.patch("/api/admin/users/uid1/tokens",
                        json={"delta": 10},
                        headers={"Authorization": f"Bearer {access_token}"})
    assert resp.status_code == 403


def test_update_role_requires_superadmin(client, admin_token):
    resp = client.patch("/api/admin/users/uid1/role",
                        json={"role": "admin"},
                        headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 403


def test_update_role_by_superadmin(client, superadmin_token, mocker):
    mocker.patch("app.routers.admin.asyncio_to_thread_update_role", return_value=None)
    resp = client.patch("/api/admin/users/uid1/role",
                        json={"role": "admin"},
                        headers={"Authorization": f"Bearer {superadmin_token}"})
    assert resp.status_code == 200
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_admin_router.py -v
```

Expected: ImportError or all FAIL

- [ ] **Step 3: Create app/routers/admin.py**

```python
# app/routers/admin.py
from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core import firestore_client
from app.middleware.auth import require_role

router = APIRouter(prefix="/api/admin", tags=["admin"])


class TokenDelta(BaseModel):
    delta: int  # positive = add, negative = deduct


class RoleUpdate(BaseModel):
    role: Literal["user", "admin"]


async def asyncio_to_thread_get_all_users():
    return await asyncio.to_thread(firestore_client.get_all_users)


async def asyncio_to_thread_update_tokens(uid: str, delta: int):
    return await asyncio.to_thread(firestore_client.update_token_balance, uid, delta)


async def asyncio_to_thread_update_role(uid: str, role: str):
    return await asyncio.to_thread(firestore_client.update_role, uid, role)


@router.get("/users")
async def list_users(admin=Depends(require_role("admin", "superadmin"))):
    users = await asyncio_to_thread_get_all_users()
    return {"users": users}


@router.patch("/users/{uid}/tokens")
async def update_tokens(
    uid: str,
    body: TokenDelta,
    admin=Depends(require_role("admin", "superadmin")),
):
    try:
        new_balance = await asyncio_to_thread_update_tokens(uid, body.delta)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"uid": uid, "new_balance": new_balance}


@router.patch("/users/{uid}/role")
async def update_role(
    uid: str,
    body: RoleUpdate,
    admin=Depends(require_role("superadmin")),
):
    try:
        await asyncio_to_thread_update_role(uid, body.role)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"uid": uid, "role": body.role}
```

- [ ] **Step 4: Add admin router to main.py**

In `app/main.py`, add:
```python
from app.routers import admin as admin_router
# ...
app.include_router(admin_router.router)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_admin_router.py -v
```

Expected: 6 PASS

- [ ] **Step 6: Commit**

```bash
git add app/routers/admin.py tests/test_admin_router.py app/main.py
git commit -m "feat(admin): add admin router with user listing, token management, and role promotion"
```

---

### Task 9: Admin UI panel

**Files:**
- Modify: `static/index.html`

- [ ] **Step 1: Add admin panel HTML**

Add a hidden admin panel div (after the existing screens):

```html
<!-- ── Admin Panel (overlay) ─────────────────────────── -->
<div id="admin-panel" style="display:none" class="admin-overlay">
  <div class="admin-modal">
    <div class="admin-header">
      <h2>Admin Dashboard</h2>
      <button onclick="hideAdminPanel()" class="btn-close">✕</button>
    </div>
    <div id="admin-content">
      <table id="users-table" class="admin-table">
        <thead>
          <tr>
            <th>Name</th><th>Email</th><th>Role</th>
            <th>Tokens</th><th>Actions</th>
          </tr>
        </thead>
        <tbody id="users-tbody"></tbody>
      </table>
    </div>
  </div>
</div>
```

- [ ] **Step 2: Add admin CSS**

```css
.admin-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.85);
  z-index: 2000; display: flex; align-items: center; justify-content: center;
}
.admin-modal {
  background: #1a1a1a; border: 1px solid #333; border-radius: 8px;
  padding: 2rem; width: 90%; max-width: 800px; max-height: 80vh;
  overflow-y: auto;
}
.admin-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; }
.admin-header h2 { color: #fff; margin: 0; font-size: 1.2rem; }
.btn-close { background: none; border: none; color: #888; font-size: 1.2rem; cursor: pointer; }
.admin-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; color: #ccc; }
.admin-table th { border-bottom: 1px solid #333; padding: 0.5rem; text-align: left; color: #888; }
.admin-table td { padding: 0.5rem; border-bottom: 1px solid #222; }
.role-badge { padding: 2px 8px; border-radius: 3px; font-size: 0.75rem; font-weight: 600; }
.role-user { background: #1a3a1a; color: #4caf50; }
.role-admin { background: #1a2a3a; color: #42a5f5; }
.role-superadmin { background: #3a1a1a; color: #ef5350; }
.token-input { width: 70px; background: #111; border: 1px solid #444; color: #fff; padding: 3px 6px; border-radius: 3px; }
.btn-sm { padding: 3px 10px; font-size: 0.8rem; border: none; border-radius: 3px; cursor: pointer; }
.btn-add { background: #1b5e20; color: #a5d6a7; }
.btn-deduct { background: #b71c1c; color: #ef9a9a; }
.btn-promote { background: #0d47a1; color: #90caf9; }
```

- [ ] **Step 3: Add admin JS functions**

```javascript
async function showAdminPanel() {
  document.getElementById('admin-panel').style.display = 'flex';
  await loadUsers();
}

function hideAdminPanel() {
  document.getElementById('admin-panel').style.display = 'none';
}

async function loadUsers() {
  const resp = await authFetch('/api/admin/users');
  if (!resp.ok) return;
  const { users } = await resp.json();
  const tbody = document.getElementById('users-tbody');
  tbody.innerHTML = users.map(u => `
    <tr>
      <td>${u.name}</td>
      <td>${u.email}</td>
      <td><span class="role-badge role-${u.role}">${u.role}</span></td>
      <td id="bal-${u.uid}">${u.token_balance}</td>
      <td>
        <input type="number" id="delta-${u.uid}" class="token-input" placeholder="±amt" />
        <button class="btn-sm btn-add" onclick="updateTokens('${u.uid}', 1)">Add</button>
        <button class="btn-sm btn-deduct" onclick="updateTokens('${u.uid}', -1)">Deduct</button>
        ${currentUser.role === 'superadmin' && u.role !== 'superadmin'
          ? `<button class="btn-sm btn-promote" onclick="toggleRole('${u.uid}', '${u.role}')">${u.role === 'admin' ? 'Demote' : 'Promote'}</button>`
          : ''}
      </td>
    </tr>
  `).join('');
}

async function updateTokens(uid, sign) {
  const raw = document.getElementById(`delta-${uid}`).value;
  const amount = Math.abs(parseInt(raw, 10));
  if (!amount || isNaN(amount)) return alert('Enter a valid amount');
  const delta = sign * amount;
  const resp = await authFetch(`/api/admin/users/${uid}/tokens`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ delta }),
  });
  if (!resp.ok) { alert('Failed to update tokens'); return; }
  const { new_balance } = await resp.json();
  document.getElementById(`bal-${uid}`).textContent = new_balance;
}

async function toggleRole(uid, currentRole) {
  const newRole = currentRole === 'admin' ? 'user' : 'admin';
  if (!confirm(`Change role to ${newRole}?`)) return;
  const resp = await authFetch(`/api/admin/users/${uid}/role`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ role: newRole }),
  });
  if (!resp.ok) { alert('Failed to update role'); return; }
  await loadUsers();
}
```

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All PASS

- [ ] **Step 5: Commit — STEP 2 COMPLETE**

```bash
git add static/index.html
git commit -m "feat(admin): create admin dashboard and role-based access control

- Admin panel UI with user table, token editor, role toggler
- Admin router: list users, add/deduct tokens, promote/demote roles
- Superadmin-only role change endpoint
- Role badges and token input with confirmation

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

> ## ⏸️ PAUSE CHECKPOINT 2
> Step 2 complete. Stop here and wait for user confirmation before proceeding to Step 3.

---

## ═══════════════════════════════════════════
## STEP 3: Tiered System & Token Logic (Backend)
## ═══════════════════════════════════════════

### Task 10: Token calculation + tier enforcement

**Files:**
- Create: `tests/test_billing.py`
- Modify: `app/routers/inference.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_billing.py`:

```python
# tests/test_billing.py
import pytest


def test_token_calculation_base_cost_only():
    from app.routers.inference import calculate_tokens
    # 0 sqm area, 0 GB file → only base cost
    assert calculate_tokens(l_sqm=0.0, s_gb=0.0) == 50


def test_token_calculation_with_area():
    from app.routers.inference import calculate_tokens
    # 10000 sqm = 1 hectare → 50 + (1 * 10) = 60
    assert calculate_tokens(l_sqm=10_000.0, s_gb=0.0) == 60


def test_token_calculation_with_file_size():
    from app.routers.inference import calculate_tokens
    # 0.5 GB → 50 + (0.5 * 200) = 150
    assert calculate_tokens(l_sqm=0.0, s_gb=0.5) == 150


def test_token_calculation_combined():
    from app.routers.inference import calculate_tokens
    # 50000 sqm (5 ha) + 1 GB → 50 + (5*10) + (1*200) = 300
    assert calculate_tokens(l_sqm=50_000.0, s_gb=1.0) == 300


def test_token_calculation_returns_int():
    from app.routers.inference import calculate_tokens
    result = calculate_tokens(l_sqm=12345.0, s_gb=0.123)
    assert isinstance(result, int)


def test_free_tier_inference_blocked_over_30mb(client, access_token, mocker):
    """Free tier (balance=0) should reject files over 30 MB."""
    mocker.patch("app.routers.inference.asyncio_to_thread_get_user", return_value={
        "uid": "uid123", "email": "t@t.com", "role": "user", "token_balance": 0,
        "daily_upload_count": 0, "last_upload_date": "",
    })

    # Create a fake 31MB file
    big_content = b"0" * (31 * 1024 * 1024)
    resp = client.post(
        "/api/inference",
        files={"file": ("test.tif", big_content, "image/tiff")},
        data={"model_name": "best_1.onnx"},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert resp.status_code == 413


def test_free_tier_daily_limit_enforced(client, access_token, mocker):
    """Free tier users who have already uploaded 3 times today get 429."""
    from datetime import date
    today = date.today().isoformat()
    mocker.patch("app.routers.inference.asyncio_to_thread_get_user", return_value={
        "uid": "uid123", "email": "t@t.com", "role": "user", "token_balance": 0,
        "daily_upload_count": 3, "last_upload_date": today,
    })
    # File check comes before daily limit check, so use a small file
    small_content = b"0" * (1024)
    resp = client.post(
        "/api/inference",
        files={"file": ("test.tif", small_content, "image/tiff")},
        data={"model_name": "best_1.onnx"},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert resp.status_code == 429


def test_commercial_tier_insufficient_tokens(client, access_token, mocker):
    """Commercial user with too few tokens gets 402."""
    from pathlib import Path
    mocker.patch("app.routers.inference.asyncio_to_thread_get_user", return_value={
        "uid": "uid123", "email": "t@t.com", "role": "user", "token_balance": 10,
        "daily_upload_count": 0, "last_upload_date": "",
    })
    mocker.patch("app.routers.inference.get_raster_area_sqm", return_value=1_000_000.0)
    # Model + YAML existence check happens before token check — mock both as present
    mocker.patch.object(Path, "exists", return_value=True)

    # cost = 50 + (100ha * 10) + ... >> 10 tokens
    small_content = b"0" * 1024
    resp = client.post(
        "/api/inference",
        files={"file": ("test.tif", small_content, "image/tiff")},
        data={"model_name": "best_1.onnx"},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert resp.status_code == 402
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_billing.py -v
```

Expected: ImportError on `calculate_tokens` — all FAIL

- [ ] **Step 3: Add billing logic to app/routers/inference.py**

Add these imports at the top of `app/routers/inference.py`:

```python
import asyncio
import math
from app.core import firestore_client
from app.core.config import settings
```

Add these two helper functions after the existing helpers (`_validate_file_id`, `_purge_old_files`):

```python
# ---------------------------------------------------------------------------
# Billing helpers
# ---------------------------------------------------------------------------

C_BASE = 50
W_AREA = 10    # per hectare
W_SIZE = 200   # per GB
FREE_TIER_MAX_BYTES = 30 * 1024 * 1024   # 30 MB
FREE_TIER_MAX_DAILY = 3


def calculate_tokens(l_sqm: float, s_gb: float) -> int:
    """Token cost formula: C_base + ((L_sqm / 10000) * W_area) + (S_gb * W_size)."""
    return math.ceil(C_BASE + (l_sqm / 10_000) * W_AREA + s_gb * W_SIZE)


def get_raster_area_sqm(tif_path: str) -> float:
    """Read raster metadata and return area in square metres (projected CRS assumed)."""
    with rasterio.open(tif_path) as src:
        transform = src.transform
        pixel_area = abs(transform.a * transform.e)   # width * height of one pixel
        total_pixels = src.width * src.height
        # If CRS is geographic (degrees), convert roughly — but prefer projected
        if src.crs and src.crs.is_geographic:
            # rough conversion: 1 degree ≈ 111_000 m
            pixel_area = pixel_area * (111_000 ** 2)
        return float(pixel_area * total_pixels)


async def asyncio_to_thread_get_user(uid: str):
    return await asyncio.to_thread(firestore_client.get_user, uid)


async def asyncio_to_thread_check_daily(uid: str):
    return await asyncio.to_thread(firestore_client.check_and_increment_daily_upload, uid)


async def asyncio_to_thread_deduct_tokens(uid: str, amount: int):
    return await asyncio.to_thread(firestore_client.deduct_tokens, uid, amount)
```

- [ ] **Step 4: Update the infer() endpoint with tier logic**

Replace the `infer()` function signature and add pre-inference checks. The function body stays the same from the inference call onwards. Add immediately after the file extension check and before the model path check:

```python
@router.post("/inference")
async def infer(
    file: UploadFile = File(...),
    model_name: str = Form("best_1.onnx"),
    tile_width: int = Form(640),
    tile_height: int = Form(640),
    min_distance: float = Form(1.0),
    conf_threshold: float = Form(0.25),
    nms_threshold: float = Form(0.4),
    current_user: dict = Depends(get_current_user),
):
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(400, "Only GeoTIFF files (.tif / .tiff) are accepted.")

    # ── Tier checks ─────────────────────────────────────────────────────
    user_data = await asyncio_to_thread_get_user(current_user["sub"])
    if not user_data:
        raise HTTPException(401, "User not found")

    token_balance = user_data.get("token_balance", 0)
    file_bytes = await file.read()
    file_size_bytes = len(file_bytes)
    file_size_gb = file_size_bytes / (1024 ** 3)

    if token_balance == 0:
        # Free tier: 30 MB limit
        if file_size_bytes > FREE_TIER_MAX_BYTES:
            raise HTTPException(413, f"Free tier limit is 30 MB. Your file is {file_size_bytes/1024/1024:.1f} MB. Add tokens to process larger files.")
        # Free tier: 3 uploads/day
        today = __import__("datetime").date.today().isoformat()
        last_date = user_data.get("last_upload_date", "")
        count = user_data.get("daily_upload_count", 0) if last_date == today else 0
        if count >= FREE_TIER_MAX_DAILY:
            raise HTTPException(429, "Free tier daily limit reached (3 uploads/day). Add tokens for unlimited access.")

    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise HTTPException(404, f"Model '{model_name}' not found. Upload it first.")
    if not YAML_PATH.exists():
        raise HTTPException(500, f"data.yaml not found at {YAML_PATH}")

    # Persist upload
    file_id = str(uuid.uuid4())
    raster_path = UPLOAD_DIR / f"{file_id}.tif"
    raster_path.write_bytes(file_bytes)   # use already-read bytes

    # ── Commercial tier: calculate and deduct tokens ─────────────────────
    tokens_deducted = 0
    if token_balance > 0:
        l_sqm = await asyncio.to_thread(get_raster_area_sqm, str(raster_path))
        cost = calculate_tokens(l_sqm=l_sqm, s_gb=file_size_gb)
        try:
            new_balance = await asyncio_to_thread_deduct_tokens(current_user["sub"], cost)
            tokens_deducted = cost
        except ValueError as e:
            raster_path.unlink(missing_ok=True)
            raise HTTPException(402, str(e))
    else:
        # Free tier: increment daily counter
        try:
            await asyncio_to_thread_check_daily(current_user["sub"])
        except ValueError:
            raster_path.unlink(missing_ok=True)
            raise HTTPException(429, "Free tier daily limit reached.")

    # ── Inference (unchanged from here) ──────────────────────────────────
    t0 = time.perf_counter()
    try:
        geojson = run_inference(
            input_tif_path=str(raster_path),
            model_path=str(model_path),
            yaml_path=str(YAML_PATH),
            tile_width=tile_width,
            tile_height=tile_height,
            min_distance=min_distance,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
        )
    except Exception as exc:
        raster_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Inference failed: {exc}") from exc

    duration = round(time.perf_counter() - t0, 2)
    geojson["metadata"]["duration_seconds"] = duration

    result_path = RESULTS_DIR / f"{file_id}.geojson"
    result_path.write_text(json.dumps(geojson, indent=2))

    return JSONResponse({
        "file_id": file_id,
        "duration_seconds": duration,
        "tokens_deducted": tokens_deducted,
        "geojson": geojson,
    })
```

- [ ] **Step 5: Run billing tests**

```bash
pytest tests/test_billing.py -v
```

Expected: `test_token_calculation_*` tests PASS. The route tests may require model files to exist — that's OK for now; they should at least hit the correct HTTP status code for the tier check (413/429/402) before reaching the model check.

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All PASS (or only model-not-found failures in route tests — acceptable since model files aren't in test env)

- [ ] **Step 7: Commit**

```bash
git add app/routers/inference.py tests/test_billing.py
git commit -m "feat(billing): implement token calculation and free tier limits

- calculate_tokens() formula: C_base + area*W_area + size*W_size
- Free tier: 30 MB max, 3 uploads/day with UTC daily reset
- Commercial tier: atomic token deduction before inference
- HTTP 413 (file too large), 429 (daily limit), 402 (insufficient tokens)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 11: Frontend tier display

**Files:**
- Modify: `static/index.html`

- [ ] **Step 1: Add token deduction feedback to upload flow**

Find the JavaScript section that handles the inference response (where `file_id` is extracted). Add token deduction display:

```javascript
// After receiving inference response:
const data = await resp.json();
if (data.tokens_deducted > 0) {
  currentUser.token_balance -= data.tokens_deducted;
  updateUserHeader();
}
```

- [ ] **Step 2: Add free tier badge to upload screen**

In the upload screen area, add informational text that updates based on tier:

```html
<div id="tier-info" class="tier-badge"></div>
```

Add JS to populate it after auth:

```javascript
function updateTierInfo() {
  const el = document.getElementById('tier-info');
  if (!el || !currentUser) return;
  if (currentUser.token_balance === 0) {
    el.innerHTML = '🆓 Free Tier &mdash; Max 30 MB &bull; 3 uploads/day &bull; <span style="color:#aaa">1 Token = IDR 1.000</span>';
    el.className = 'tier-badge tier-free';
  } else {
    el.innerHTML = `💎 Commercial &mdash; ${currentUser.token_balance.toLocaleString()} tokens available &bull; <span style="color:#aaa">1 Token = IDR 1.000</span>`;
    el.className = 'tier-badge tier-commercial';
  }
}
```

Call `updateTierInfo()` inside `onAuthSuccess()`.

Add CSS:
```css
.tier-badge { font-size: 0.8rem; padding: 0.4rem 0.8rem; border-radius: 4px; margin-bottom: 1rem; }
.tier-free { background: #1a2a1a; color: #81c784; border: 1px solid #2e4d2e; }
.tier-commercial { background: #1a1a2a; color: #90caf9; border: 1px solid #2e3d5e; }
```

- [ ] **Step 3: Commit — STEP 3 COMPLETE**

```bash
git add static/index.html
git commit -m "feat(billing): implement token calculation and free tier limits

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

> ## ⏸️ PAUSE CHECKPOINT 3
> Step 3 complete. Stop here and wait for user confirmation before proceeding to Step 4.

---

## ═══════════════════════════════════════════════════
## STEP 4: Commercial GPU Architecture (Infrastructure)
## ═══════════════════════════════════════════════════

### Task 12: Presign + submit endpoints for large files

**Files:**
- Modify: `app/routers/inference.py`

- [ ] **Step 1: Write tests**

Add to `tests/test_billing.py`:

```python
def test_presign_requires_commercial_tier(client, access_token, mocker):
    """Free tier (balance=0) cannot use presigned upload."""
    mocker.patch("app.routers.inference.asyncio_to_thread_get_user", return_value={
        "uid": "uid123", "email": "t@t.com", "role": "user", "token_balance": 0,
        "daily_upload_count": 0, "last_upload_date": "",
    })
    resp = client.post(
        "/api/inference/presign",
        json={"filename": "large.tif", "file_size_bytes": 1024 * 1024 * 100},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert resp.status_code == 403


def test_presign_returns_upload_url(client, access_token, mocker):
    mocker.patch("app.routers.inference.asyncio_to_thread_get_user", return_value={
        "uid": "uid123", "email": "t@t.com", "role": "user", "token_balance": 5000,
        "daily_upload_count": 0, "last_upload_date": "",
    })
    mocker.patch("app.routers.inference.generate_signed_upload_url",
                 return_value=("https://storage.googleapis.com/signed", "uploads/uid123/file.tif"))
    resp = client.post(
        "/api/inference/presign",
        json={"filename": "large.tif", "file_size_bytes": 1024 * 1024 * 100},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "upload_url" in data
    assert "gcs_path" in data
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_billing.py::test_presign_requires_commercial_tier tests/test_billing.py::test_presign_returns_upload_url -v
```

Expected: ImportError or FAIL

- [ ] **Step 3: Add GCS helpers and presign/submit endpoints to inference.py**

Add import at top:
```python
from google.cloud import storage as gcs
```

Add helper function after the existing billing helpers:

```python
def generate_signed_upload_url(user_uid: str, filename: str) -> tuple[str, str]:
    """Generate a GCS signed PUT URL valid for 1 hour. Returns (url, gcs_path)."""
    client = gcs.Client(project=settings.firestore_project_id)
    bucket = client.bucket(settings.gcs_bucket_name)
    safe_name = "".join(c for c in filename if c.isalnum() or c in ("_", "-", "."))
    gcs_path = f"uploads/{user_uid}/{uuid.uuid4().hex}_{safe_name}"
    blob = bucket.blob(gcs_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=__import__("datetime").timedelta(hours=1),
        method="PUT",
        content_type="image/tiff",
    )
    return url, gcs_path
```

Add the two new endpoints (before the cleanup endpoint):

```python
class PresignRequest(BaseModel):
    filename: str
    file_size_bytes: int


class SubmitRequest(BaseModel):
    gcs_path: str
    model_name: str = "best_1.onnx"
    tile_width: int = 640
    tile_height: int = 640
    min_distance: float = 1.0
    conf_threshold: float = 0.25
    nms_threshold: float = 0.4


@router.post("/inference/presign")
async def presign_upload(
    body: PresignRequest,
    current_user: dict = Depends(get_current_user),
):
    """Commercial tier only: get a GCS signed URL for direct large-file upload."""
    user_data = await asyncio_to_thread_get_user(current_user["sub"])
    if not user_data or user_data.get("token_balance", 0) == 0:
        raise HTTPException(403, "Presigned upload is for commercial tier only (requires token balance > 0)")

    try:
        upload_url, gcs_path = await asyncio.to_thread(
            generate_signed_upload_url, current_user["sub"], body.filename
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to generate signed URL: {e}")

    return {"upload_url": upload_url, "gcs_path": gcs_path}


@router.post("/inference/submit")
async def submit_gcs_inference(
    body: SubmitRequest,
    current_user: dict = Depends(get_current_user),
):
    """Commercial tier: run inference on a file already uploaded to GCS."""
    user_data = await asyncio_to_thread_get_user(current_user["sub"])
    if not user_data or user_data.get("token_balance", 0) == 0:
        raise HTTPException(403, "Submit endpoint is for commercial tier only")

    # Forward to GPU worker
    if not settings.gpu_worker_url:
        raise HTTPException(503, "GPU worker not configured")

    import httpx as _httpx
    async with _httpx.AsyncClient(timeout=3600) as http:
        try:
            gpu_resp = await http.post(
                f"{settings.gpu_worker_url}/api/inference/internal",
                json={
                    "gcs_path": body.gcs_path,
                    "model_name": body.model_name,
                    "tile_width": body.tile_width,
                    "tile_height": body.tile_height,
                    "min_distance": body.min_distance,
                    "conf_threshold": body.conf_threshold,
                    "nms_threshold": body.nms_threshold,
                    "user_uid": current_user["sub"],
                },
                headers={"X-Internal-Secret": settings.cleanup_secret},
            )
        except Exception as e:
            raise HTTPException(502, f"GPU worker unreachable: {e}")

    if gpu_resp.status_code != 200:
        raise HTTPException(gpu_resp.status_code, f"GPU worker error: {gpu_resp.text}")

    result = gpu_resp.json()

    # Deduct tokens based on actual file stats returned by GPU worker
    l_sqm = result.get("area_sqm", 0.0)
    s_gb = result.get("file_size_gb", 0.0)
    cost = calculate_tokens(l_sqm=l_sqm, s_gb=s_gb)
    try:
        await asyncio_to_thread_deduct_tokens(current_user["sub"], cost)
    except ValueError as e:
        raise HTTPException(402, str(e))

    return JSONResponse({**result, "tokens_deducted": cost})
```

Add `from pydantic import BaseModel` to the imports at the top of `app/routers/inference.py` (alongside existing FastAPI imports).

- [ ] **Step 4: Run billing tests**

```bash
pytest tests/test_billing.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add app/routers/inference.py tests/test_billing.py
git commit -m "feat(infra): add GCS presign and GPU submit endpoints for large commercial files"
```

---

### Task 13: Infra scripts

**Files:**
- Create: `infra/deploy-cpu.sh`
- Create: `infra/deploy-gpu.sh`
- Create: `infra/setup-gpu-sa.sh`

- [ ] **Step 1: Create infra/deploy-cpu.sh**

```bash
mkdir -p infra
```

Create `infra/deploy-cpu.sh`:

```bash
#!/usr/bin/env bash
# deploy-cpu.sh — update CPU Cloud Run service with new env vars
# Usage: PROJECT_ID=palmcounting bash infra/deploy-cpu.sh

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-palmcounting}"
REGION="asia-southeast1"
SERVICE="palm-counter"
IMAGE="asia-southeast1-docker.pkg.dev/${PROJECT_ID}/palm-counting-repo/palm-counter:latest"

echo "Updating CPU service: ${SERVICE}"

gcloud run services update "${SERVICE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --update-env-vars="\
GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID},\
GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET},\
JWT_SECRET=${JWT_SECRET},\
JWT_ACCESS_EXPIRE_MINUTES=${JWT_ACCESS_EXPIRE_MINUTES:-15},\
JWT_REFRESH_EXPIRE_DAYS=${JWT_REFRESH_EXPIRE_DAYS:-7},\
FIRESTORE_PROJECT_ID=${PROJECT_ID},\
FRONTEND_URL=https://palm-counter-981519809891.asia-southeast1.run.app,\
GCS_BUCKET_NAME=${GCS_BUCKET_NAME:-palmcounting-uploads},\
GPU_WORKER_URL=${GPU_WORKER_URL:-},\
PYTHONUNBUFFERED=1"

echo "CPU service updated."
```

- [ ] **Step 2: Create infra/deploy-gpu.sh**

Create `infra/deploy-gpu.sh`:

```bash
#!/usr/bin/env bash
# deploy-gpu.sh — deploy GPU Cloud Run service for commercial tier
# Usage: PROJECT_ID=palmcounting bash infra/deploy-gpu.sh
#
# Prerequisites:
#   - GPU quota approved for asia-southeast1 (nvidia-l4)
#   - GPU Cloud Run service image built and pushed
#   - Service account with Storage + Firestore access (see setup-gpu-sa.sh)

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-palmcounting}"
REGION="asia-southeast1"
SERVICE="palm-counter-gpu"
IMAGE="asia-southeast1-docker.pkg.dev/${PROJECT_ID}/palm-counting-repo/palm-counter-gpu:latest"
SA="palm-counter-gpu@${PROJECT_ID}.iam.gserviceaccount.com"

echo "Deploying GPU service: ${SERVICE}"

gcloud run deploy "${SERVICE}" \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --platform=managed \
  --no-allow-unauthenticated \
  --service-account="${SA}" \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --memory=32Gi \
  --cpu=8 \
  --min-instances=0 \
  --max-instances=3 \
  --timeout=3600 \
  --set-env-vars="\
FIRESTORE_PROJECT_ID=${PROJECT_ID},\
GCS_BUCKET_NAME=${GCS_BUCKET_NAME:-palmcounting-uploads},\
CLEANUP_SECRET=${CLEANUP_SECRET:-},\
PYTHONUNBUFFERED=1"

echo "GPU service deployed."
GPU_URL=$(gcloud run services describe "${SERVICE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status.url)")
echo "GPU service URL: ${GPU_URL}"
echo "Add this as GPU_WORKER_URL in your CPU service env vars."
```

- [ ] **Step 3: Create infra/setup-gpu-sa.sh**

Create `infra/setup-gpu-sa.sh`:

```bash
#!/usr/bin/env bash
# setup-gpu-sa.sh — create and configure service account for GPU Cloud Run
# Usage: PROJECT_ID=palmcounting bash infra/setup-gpu-sa.sh

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-palmcounting}"
SA_NAME="palm-counter-gpu"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "Creating service account: ${SA_EMAIL}"

gcloud iam service-accounts create "${SA_NAME}" \
  --display-name="Palm Counter GPU Worker" \
  --project="${PROJECT_ID}" || echo "Service account already exists"

# Allow GPU service to read/write GCS bucket
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin"

# Allow GPU service to read/write Firestore
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/datastore.user"

echo "Service account configured."
echo "Next: run deploy-gpu.sh to deploy the GPU Cloud Run service."
```

Make scripts executable:
```bash
chmod +x infra/deploy-cpu.sh infra/deploy-gpu.sh infra/setup-gpu-sa.sh
```

- [ ] **Step 4: Commit**

```bash
git add infra/
git commit -m "chore(infra): add GPU Cloud Run deployment and service account setup scripts"
```

---

### Task 14: Update CI/CD to pass new env vars to Cloud Run

**Files:**
- Modify: `.github/workflows/deploy.yml`
- Modify: `cloudbuild.yaml`

- [ ] **Step 1: Update cloudbuild.yaml to pass env vars on deploy**

In `cloudbuild.yaml`, replace the `cloud-run-deploy` step's `--set-env-vars` line:

```yaml
  # ── Step 4: Deploy to Cloud Run ──────────────────────────────────────
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    id: 'cloud-run-deploy'
    waitFor: ['docker-push']
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'palm-counter'
      - '--image=asia-southeast1-docker.pkg.dev/$PROJECT_ID/palm-counting-repo/palm-counter:$COMMIT_SHA'
      - '--region=asia-southeast1'
      - '--platform=managed'
      - '--allow-unauthenticated'
      - '--memory=2Gi'
      - '--cpu=2'
      - '--min-instances=0'
      - '--max-instances=5'
      - '--timeout=300'
      - >-
        --update-env-vars=PYTHONUNBUFFERED=1,FIRESTORE_PROJECT_ID=$PROJECT_ID,FRONTEND_URL=https://palm-counter-981519809891.asia-southeast1.run.app
      - '--quiet'
```

Note: Secrets (CLIENT_ID, CLIENT_SECRET, JWT_SECRET) must be set via Cloud Run Secret Manager or directly via `gcloud run services update` after first deploy — **do not put secrets in cloudbuild.yaml**.

- [ ] **Step 2: Update GitHub Actions to add a secrets-reminder step**

In `.github/workflows/deploy.yml`, add a new step after the deployment step:

```yaml
      # ── 5. Remind about secrets ────────────────────────────────
      - name: 🔑 Post-deploy secrets reminder
        run: |
          echo "============================================="
          echo "⚠️  REMINDER: Set these secrets via Cloud Run if not already done:"
          echo "   GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, JWT_SECRET"
          echo "   Run: bash infra/deploy-cpu.sh  (with env vars set)"
          echo "============================================="
```

- [ ] **Step 3: Run full test suite one final time**

```bash
pytest tests/ -v
```

Expected: All PASS

- [ ] **Step 4: Commit — STEP 4 COMPLETE**

```bash
git add cloudbuild.yaml .github/workflows/deploy.yml
git commit -m "chore(infra): add GPU architecture configuration for commercial tier

- GCS presigned URL flow for files up to 10 GB (bypasses Cloud Run limit)
- GPU Cloud Run service deployment script (NVIDIA L4, 32GB RAM)
- Service account setup script with GCS + Firestore permissions
- Updated CI/CD with post-deploy secrets reminder
- Commercial requests routed to GPU worker via internal HTTP

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

> ## ⏸️ PAUSE CHECKPOINT 4
> All 4 steps complete. CI/CD pipeline is intact. Ready for Payment Gateway integration in a future step.

---

## GitHub Secrets to Add

After merging, add these to GitHub Repository → Settings → Secrets → Actions:

| Secret Name | Value |
|---|---|
| `GOOGLE_CLIENT_ID` | From `.env` |
| `GOOGLE_CLIENT_SECRET` | From `.env` |
| `JWT_SECRET` | From `.env` |
| `GCS_BUCKET_NAME` | `palmcounting-uploads` |

Then run `bash infra/deploy-cpu.sh` (with env vars exported) to update the live Cloud Run service with the new secrets.

---

## Post-Step-4 Verification Checklist

- [ ] `pytest tests/ -v` — all pass
- [ ] `git push origin main` — CI/CD pipeline green
- [ ] Login flow works end-to-end locally (`uvicorn app.main:app --port 8080`)
- [ ] `fakmalpradana@gmail.com` gets `superadmin` role automatically
- [ ] Admin panel visible to admin/superadmin only
- [ ] Free tier: 30MB limit returns 413, 4th upload returns 429
- [ ] Commercial tier: insufficient tokens returns 402
- [ ] `/api/cleanup` still works without auth (uses CLEANUP_SECRET)
