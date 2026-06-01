# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Palm Counter is a SaaS geospatial application that detects palm trees in GeoTIFF satellite imagery using YOLO ONNX inference. It tiles large rasters, runs detection on each tile, deduplicates overlapping detections via R-tree spatial indexing, and returns results as GeoJSON. It also supports land-cover classification as a second task type.

## Commands

```bash
# Run dev server (requires conda env "palmcounting" activated)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or use the convenience script
bash run.sh

# Run all tests
pytest

# Run a single test file
pytest tests/test_auth_router.py

# Run a single test
pytest tests/test_auth_router.py::test_function_name -v
```

## Architecture

### Two Entry Points
- **`main.py`** (root) — standalone CLI script for local batch processing (tile + detect + shapefile output). Not used by the web app.
- **`app/main.py`** — FastAPI application serving the REST API and static frontend.

### Backend (`app/`)
- **`app/core/config.py`** — `pydantic-settings` config loaded from `.env` / `.env.local`. Set `DEV_MODE=true` for local dev (bypasses Google OAuth and Firestore with a mock superadmin user).
- **`app/core/inference.py`** — Core detection logic: tiling, YOLO ONNX inference, geo-coordinate math, overlap filtering. Auto-selects backend: onnxruntime-gpu > onnxruntime (CPU) > cv2.dnn fallback.
- **`app/core/land_cover_inference.py`** — Land-cover classification pipeline (separate from palm detection).
- **`app/core/firestore_client.py`** — All Firestore operations: user CRUD, token management (transactional), job CRUD, admin stats. Also contains dev-mode mock implementations. Now includes `create_email_user`, `get_user_by_email`, `update_user_password`, `set_email_verified` for email/password auth.
- **`app/core/auth.py`** — JWT creation/verification (HS256 via python-jose). Google OAuth URL builder. Includes `create_email_token()` for short-lived email verify / password reset tokens.
- **`app/core/email.py`** — SMTP email sender (supports STARTTLS port 587 and implicit SSL port 465). Sends verification and password-reset emails. No-op if `SMTP_HOST` is not configured.
- **`app/core/password.py`** — bcrypt password hashing and verification helpers.
- **`app/middleware/auth.py`** — FastAPI dependencies: `get_current_user` (extracts JWT from Bearer header), `require_role()` factory for role-based access (user/admin/superadmin).
- **`app/routers/`** — Route handlers: `auth.py` (Google OAuth + email/password auth), `inference.py` (direct inference), `jobs.py` (job lifecycle with approval/billing), `admin.py` (admin dashboard).

### Authentication Methods
Two auth paths share the same JWT token format and refresh-cookie mechanism:
- **Google OAuth** — `/api/auth/login` → Google → `/api/auth/callback` → set refresh cookie.
- **Email/Password** — register (`POST /api/auth/register`) → verify email link → login (`POST /api/auth/email-login`). Password reset via `/api/auth/request-password-reset` + `/api/auth/reset-password`. Resend verification via `/api/auth/resend-verification`.
- **Account linking** — if a Google-authed user registers with the same email, the password is added to the existing account (no duplicate user).
- **Dev mode** — `DEV_MODE=true` adds an in-memory user store so email/password flows can be tested locally without Firestore or a real SMTP server.

### Billing / Job Flow
Jobs follow a lifecycle: `uploading → approval → processing → done/failed`. The `/api/jobs/{id}/approve` endpoint atomically deducts tokens before starting background inference. Users get one free trial per task type (palm counting and land cover).

### Frontend
Single-page app at `static/index.html` — plain HTML/JS with Leaflet map. No build step required.

### Data Stores
- **Google Cloud Firestore** — users collection (auth, roles, token balances) and jobs collection. Bypassed entirely in dev mode.
- **Google Cloud Storage** — file uploads/results in production. Local `uploads/` and `results/` directories used in dev.

### Deployment
Deployed to GCP Cloud Run via Cloud Build (`cloudbuild.yaml`). Model weights are fetched from GCS during build and baked into the Docker image. GPU variant uses `Dockerfile.gpu` + `cloudbuild-gpu.yaml`.

## Testing

Tests use `pytest` + `pytest-asyncio` + `pytest-mock`. Firestore is auto-mocked via `conftest.py` (the `mock_firestore_db` fixture is `autouse=True`). The conftest forces `DEV_MODE=false` and sets dummy env vars before any app imports — this is intentional so tests exercise the real auth/Firestore code paths with mocks rather than dev-mode shortcuts.

## Key Env Vars

All configured via `app/core/config.py` (loaded from `.env` / `.env.local`):

| Variable | Purpose |
|---|---|
| `DEV_MODE` | `true` bypasses OAuth + Firestore with mock superadmin |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Google OAuth credentials |
| `JWT_SECRET` | Signing key for access/refresh tokens |
| `FIRESTORE_PROJECT_ID` | GCP project for Firestore |
| `GCS_BUCKET_NAME` | Cloud Storage bucket for uploads/results |
| `GPU_WORKER_URL` | URL of GPU Cloud Run service (for offloading inference) |
| `FRONTEND_URL` | Used for CORS and OAuth redirect URI |
| `SMTP_HOST` | SMTP server hostname (leave empty to disable email sending) |
| `SMTP_PORT` | SMTP port — 587 for STARTTLS (default), 465 for implicit SSL |
| `SMTP_USER` | SMTP login username |
| `SMTP_PASSWORD` | SMTP login password |
| `SMTP_FROM_EMAIL` | Sender address (falls back to `SMTP_USER` if unset) |
| `SMTP_USE_TLS` | `true` to use STARTTLS (default) |
| `SMTP_USE_SSL` | `true` for implicit SSL on port 465 (overrides TLS) |
