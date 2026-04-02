# SaaS Auth, Billing & GPU Architecture — Design Spec
**Date:** 2026-04-02  
**Project:** Palm Counter (WebGIS YOLO)  
**Status:** Approved

---

## 1. Overview

This spec covers transitioning Palm Counter from an open, unauthenticated API into a SaaS platform with:
- Google OAuth 2.0 login (Authorization Code Flow)
- Firestore-backed user management
- Role-based access control (user / admin / superadmin)
- Tiered billing (Free vs Commercial) with token deduction
- Separate CPU (free) and GPU (commercial) Cloud Run workers
- Signed URL strategy for large commercial uploads (up to 10 GB)

---

## 2. Tech Stack Additions

| Concern | Choice | Reason |
|---|---|---|
| Database | Firestore (Native Mode) | Serverless, no connection pooling, fits flat user schema |
| Auth flow | OAuth 2.0 Authorization Code Flow | Client secret stays on server, industry standard |
| Session | Access token in JS memory + Refresh token in httpOnly cookie | XSS-safe, CSRF-safe, stateless Cloud Run compatible |
| JWT signing | HS256, `python-jose` | Simple, fast, standard |
| OAuth client | `httpx` (async HTTP) | Already compatible with FastAPI async |
| Firestore client | `google-cloud-firestore` | Official GCP SDK |

---

## 3. Firestore Schema

**Collection:** `users`  
**Document ID:** Google UID (`sub` field from Google ID token)

```
users/{google_uid}
  ├── email                string   required
  ├── name                 string   from Google profile
  ├── avatar               string   Google profile picture URL
  ├── role                 string   "user" | "admin" | "superadmin"
  ├── token_balance        int      default: 0
  ├── daily_upload_count   int      default: 0
  └── last_upload_date     string   ISO date "YYYY-MM-DD", UTC
```

**Superadmin rule:** If `email == "fakmalpradana@gmail.com"`, `role` is always forced to `"superadmin"` on every upsert — never overridable via API.

---

## 4. Authentication Flow

### 4.1 Login

```
GET /api/auth/login
  → 302 redirect to Google OAuth URL
  → user authenticates with Google
  → Google redirects to GET /api/auth/callback?code=xxx&state=xxx
  → backend exchanges code for tokens via Google token endpoint
  → backend upserts user in Firestore
  → sets refresh_token as httpOnly cookie (7 days)
  → redirects to frontend with access_token in URL fragment (#token=xxx)
  → frontend strips token from URL, stores in JS memory variable
```

### 4.2 Token Strategy

| Token | Lifetime | Storage | Usage |
|---|---|---|---|
| Access token | 15 min | JS memory variable | `Authorization: Bearer <token>` on every API call |
| Refresh token | 7 days | `httpOnly; Secure; SameSite=Strict` cookie | Sent automatically to `POST /api/auth/refresh` |

### 4.3 Silent Refresh

On every page load, frontend immediately calls `POST /api/auth/refresh`. If the cookie is valid, a new access token is returned and stored in memory. If the cookie is expired or absent, user is shown the login screen.

### 4.4 Logout

`POST /api/auth/logout` — clears the httpOnly cookie server-side, frontend discards the in-memory access token.

---

## 5. New Backend Files

```
app/
├── routers/
│   ├── inference.py         (existing — add auth dependency, tier routing)
│   ├── auth.py              (NEW)
│   └── admin.py             (NEW)
├── core/
│   ├── inference.py         (existing)
│   ├── auth.py              (NEW — JWT utils, OAuth helpers)
│   └── firestore_client.py  (NEW — user CRUD)
├── middleware/
│   └── auth.py              (NEW — get_current_user dependency)
└── main.py                  (update — add routers, update CORS)
```

---

## 6. API Endpoints

### Auth Router (`/api/auth`)

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/api/auth/login` | No | Redirect to Google OAuth |
| GET | `/api/auth/callback` | No | Handle OAuth callback, issue tokens |
| POST | `/api/auth/refresh` | Cookie | Issue new access token |
| GET | `/api/auth/me` | Bearer | Return current user info |
| POST | `/api/auth/logout` | Bearer | Clear refresh token cookie |

### Admin Router (`/api/admin`)

| Method | Path | Min Role | Description |
|---|---|---|---|
| GET | `/api/admin/users` | admin | List all users |
| PATCH | `/api/admin/users/{uid}/tokens` | admin | Add/deduct tokens `{ "delta": int }` |
| PATCH | `/api/admin/users/{uid}/role` | superadmin | Change role `{ "role": "admin"\|"user" }` |

### Inference Router (updated)

| Method | Path | Min Role | Notes |
|---|---|---|---|
| POST | `/api/inference` | user | Tier check before processing |
| POST | `/api/inference/presign` | user (commercial) | Returns GCS signed URL for large upload |
| POST | `/api/inference/submit` | user (commercial) | Submit job after GCS upload |
| GET | `/api/download/{id}` | user | Unchanged |
| GET | `/api/preview/{id}` | user | Unchanged |
| GET | `/api/models` | user | Unchanged |
| POST | `/api/models` | user | Unchanged |
| GET | `/api/cleanup` | No (secret) | Unchanged |

---

## 7. Tier Logic

### 7.1 Free Tier (token_balance == 0)

- Max file size: **30 MB**
- Max uploads per day: **3** (UTC-based daily reset)
- Routed to: existing CPU Cloud Run service
- Daily reset logic:
  ```python
  if user.last_upload_date != today_utc:
      user.daily_upload_count = 0
      user.last_upload_date = today_utc
  if user.daily_upload_count >= 3:
      raise HTTP 429
  ```

### 7.2 Commercial Tier (token_balance > 0)

Token cost calculated before inference:

```
Total_K = C_base + ((L_sqm / 10000) * W_area) + (S_gb * W_size)
```

| Constant | Value |
|---|---|
| C_base | 50 |
| W_area | 10 (per hectare) |
| W_size | 200 (per GB) |

- `L_sqm` — raster area in m², read from file metadata via rasterio
- `S_gb` — upload file size in GB

**Pre-inference flow:**
1. Read metadata → compute `Total_K`
2. If `token_balance < Total_K` → HTTP 402 "Insufficient tokens"
3. Atomically deduct `Total_K` (Firestore transaction)
4. Run inference on GPU worker
5. Return result + updated `token_balance`

**UI note:** Display `1 Token = IDR 1,000` as static info text. No payment integration yet.

---

## 8. GPU Architecture

### 8.1 Two-Service Design

```
palm-counter (main — existing, IS the CPU worker)
  CPU: 2, RAM: 2GB, Cloud Run managed
  Handles: auth, routing, billing, admin
  Free tier inference runs directly in this service (no separate CPU worker)
  Commercial inference forwarded to palm-counter-gpu via internal HTTP

palm-counter-gpu (new — Step 4 only)
  GPU: NVIDIA L4 (1x), CPU: 8, RAM: 32GB
  No public access (unauthenticated access disabled)
  Called only by palm-counter main via service-to-service auth
  Timeout: 3600s
  Max instances: 3
```

### 8.2 Large File Strategy (Commercial, up to 10 GB)

Cloud Run has a 32MB HTTP body limit. Commercial users bypass this:

```
1. POST /api/inference/presign
   → backend generates GCS signed URL (PUT, 1hr expiry)
   → returns { upload_url, gcs_path }

2. Frontend PUT file directly to GCS signed URL (no backend involved)

3. POST /api/inference/submit { gcs_path }
   → backend deducts tokens
   → GPU worker reads file from GCS directly
   → result written to GCS
   → returns download URL
```

### 8.3 Infrastructure Files

```
infra/
├── deploy-cpu.sh      Update existing Cloud Run CPU service config
├── deploy-gpu.sh      Deploy GPU Cloud Run service
└── setup-gpu-sa.sh    Grant GPU service account GCS + Firestore access
```

---

## 9. Environment Variables

| Variable | Required By | Description |
|---|---|---|
| `GOOGLE_CLIENT_ID` | Backend | OAuth 2.0 Client ID |
| `GOOGLE_CLIENT_SECRET` | Backend | OAuth 2.0 Client Secret |
| `JWT_SECRET` | Backend | Random 32-byte secret for HS256 signing |
| `JWT_ACCESS_EXPIRE_MINUTES` | Backend | Default: 15 |
| `JWT_REFRESH_EXPIRE_DAYS` | Backend | Default: 7 |
| `FIRESTORE_PROJECT_ID` | Backend | GCP project ID |
| `GPU_WORKER_URL` | Backend | Internal URL of GPU Cloud Run service |
| `GCS_BUCKET_NAME` | Backend | Bucket for large commercial uploads |
| `FRONTEND_URL` | Backend | For OAuth redirect and CORS (`https://palm-counter-...run.app`) |
| `CLEANUP_MAX_AGE_HOURS` | Backend | Existing — unchanged |
| `CLEANUP_SECRET` | Backend | Existing — unchanged |

---

## 10. Frontend Changes

All changes to `static/index.html`:

1. **Auth state management** — JS module-level variable `let accessToken = null`
2. **Login screen** — shown when no valid token; "Sign in with Google" button → GET `/api/auth/login`
3. **Silent refresh** — on page load, call `POST /api/auth/refresh`; on success store token + show app; on fail show login screen
4. **Auth header injection** — all `fetch()` calls add `Authorization: Bearer ${accessToken}`
5. **Token expiry handling** — on 401 response, attempt one silent refresh then re-login
6. **Token balance display** — show current balance in header; show `1 Token = IDR 1,000` info
7. **Admin panel** — hidden section, visible only when `role == "admin"` or `"superadmin"`

---

## 11. Implementation Order (Steps)

1. **Step 1** — Firestore client + Google OAuth + JWT auth + user upsert + protect existing routes
2. **Step 2** — Admin router + admin dashboard UI + role enforcement
3. **Step 3** — Tier logic + token calculation + free tier limits + request router
4. **Step 4** — GPU Cloud Run service + signed URL flow + infra scripts

Each step commits independently. No payment gateway in scope.
