# app/core/firestore_client.py
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Any, Literal, Optional

from app.core.config import settings

SUPERADMIN_EMAIL = "fakmalpradana@gmail.com"

# ── Dev-mode mock user (used when DEV_MODE=true, Firestore is never contacted) ──
_DEV_USER: dict = {
    "uid":                  "dev-local-user",
    "email":                "dev@local.test",
    "name":                 "Dev User",
    "avatar":               "",
    "role":                 "superadmin",
    "token_balance":        9999,
    "daily_upload_count":   0,
    "last_upload_date":     "",
    "free_palm_used":       False,
    "free_landcover_used":  False,
}

_db = None


def get_db():
    global _db
    if _db is None:
        from google.cloud import firestore
        _db = firestore.Client(project=settings.firestore_project_id)
    return _db


def upsert_user(google_uid: str, email: str, name: str, avatar: str) -> dict:
    if settings.dev_mode:
        return _DEV_USER
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
            "email":               email,
            "name":                name,
            "avatar":              avatar,
            "role":                role,
            "token_balance":       0,
            "daily_upload_count":  0,
            "last_upload_date":    "",
            "free_palm_used":      False,
            "free_landcover_used": False,
        }
        ref.set(data)
        return {**data, "uid": google_uid}


def get_user(google_uid: str) -> Optional[dict]:
    if settings.dev_mode:
        return _DEV_USER
    db = get_db()
    doc = db.collection("users").document(google_uid).get()
    if not doc.exists:
        return None
    return {**doc.to_dict(), "uid": google_uid}


def get_all_users() -> list[dict]:
    if settings.dev_mode:
        return [_DEV_USER]
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
    if settings.dev_mode:
        return max(0, _DEV_USER["token_balance"] - amount)
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


def mark_palm_free_used(google_uid: str) -> None:
    """Atomically claim the one-time free palm-counting trial.

    Raises ValueError if the trial has already been used.
    No-op in dev mode (dev user has tokens, so free-tier path is never reached).
    """
    if settings.dev_mode:
        return
    from google.cloud import firestore

    db = get_db()
    ref = db.collection("users").document(google_uid)

    @firestore.transactional
    def _txn(transaction, ref):
        snap = ref.get(transaction=transaction)
        if not snap.exists:
            raise ValueError("User not found")
        if snap.to_dict().get("free_palm_used", False):
            raise ValueError("Free palm-counting trial already used")
        transaction.update(ref, {"free_palm_used": True})

    _txn(db.transaction(), ref)


def mark_landcover_free_used(google_uid: str) -> None:
    """Atomically claim the one-time free land-cover trial.

    Raises ValueError if the trial has already been used.
    """
    if settings.dev_mode:
        return
    from google.cloud import firestore

    db = get_db()
    ref = db.collection("users").document(google_uid)

    @firestore.transactional
    def _txn(transaction, ref):
        snap = ref.get(transaction=transaction)
        if not snap.exists:
            raise ValueError("User not found")
        if snap.to_dict().get("free_landcover_used", False):
            raise ValueError("Free land-cover trial already used")
        transaction.update(ref, {"free_landcover_used": True})

    _txn(db.transaction(), ref)


def check_and_increment_daily_upload(google_uid: str) -> dict:
    """Reset counter if date changed, increment, return updated counts.
    Raises ValueError if daily limit (3) is reached.
    """
    if settings.dev_mode:
        return {"daily_upload_count": 1, "last_upload_date": date.today().isoformat()}
    from google.cloud import firestore

    db = get_db()
    ref = db.collection("users").document(google_uid)
    today = date.today().isoformat()

    @firestore.transactional
    def _txn(transaction, ref):
        snap = ref.get(transaction=transaction)
        if not snap.exists:
            raise ValueError("User not found")
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


# ── Job Management ─────────────────────────────────────────────────────────────

JobStatus = Literal["uploading", "approval", "processing", "done", "failed"]
TaskType = Literal["palm_counting", "land_cover"]

_DEV_JOBS: list[dict] = []


def create_job(
    user_email: str,
    job_name: str,
    task_type: TaskType,
    parameters: dict[str, Any],
    file_uri: str = "",
) -> dict:
    """Create a new job document in Firestore. Returns the full job dict."""
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    job: dict[str, Any] = {
        "job_id":     job_id,
        "user_email": user_email,
        "job_name":   job_name or "Untitled Job",
        "task_type":  task_type,
        "parameters": parameters,
        "status":     "uploading",
        "file_uri":   file_uri,
        "result_uri": "",
        "token_cost": 0,
        "created_at": now.isoformat(),
    }
    if settings.dev_mode:
        _DEV_JOBS.append(job)
        return job
    db = get_db()
    db.collection("jobs").document(job_id).set(job)
    return job


def get_jobs_for_user(user_email: str) -> list[dict]:
    """Return all jobs belonging to *user_email*, newest first."""
    if settings.dev_mode:
        return sorted(
            [j for j in _DEV_JOBS if j["user_email"] == user_email],
            key=lambda j: j["created_at"],
            reverse=True,
        )
    db = get_db()
    docs = (
        db.collection("jobs")
        .where("user_email", "==", user_email)
        .order_by("created_at", direction="DESCENDING")
        .stream()
    )
    return [doc.to_dict() for doc in docs]


def get_all_jobs() -> list[dict]:
    """Return every job across all users (admin use), newest first."""
    if settings.dev_mode:
        return sorted(_DEV_JOBS, key=lambda j: j["created_at"], reverse=True)
    db = get_db()
    docs = (
        db.collection("jobs")
        .order_by("created_at", direction="DESCENDING")
        .limit(200)
        .stream()
    )
    return [doc.to_dict() for doc in docs]


def get_job(job_id: str) -> Optional[dict]:
    """Fetch a single job by ID."""
    if settings.dev_mode:
        return next((j for j in _DEV_JOBS if j["job_id"] == job_id), None)
    db = get_db()
    doc = db.collection("jobs").document(job_id).get()
    return doc.to_dict() if doc.exists else None


def update_job(job_id: str, updates: dict[str, Any]) -> Optional[dict]:
    """Partially update a job document. Returns the updated job or None if not found."""
    if settings.dev_mode:
        job = next((j for j in _DEV_JOBS if j["job_id"] == job_id), None)
        if job is None:
            return None
        job.update(updates)
        return job
    db = get_db()
    ref = db.collection("jobs").document(job_id)
    if not ref.get().exists:
        return None
    ref.update(updates)
    return ref.get().to_dict()


def get_admin_stats() -> dict:
    """Return global KPI counts for the admin dashboard."""
    if settings.dev_mode:
        total_tokens = sum(j.get("token_cost", 0) for j in _DEV_JOBS)
        return {
            "total_users":   1,
            "total_jobs":    len(_DEV_JOBS),
            "total_tokens":  total_tokens,
        }
    db = get_db()
    users = list(db.collection("users").stream())
    jobs  = list(db.collection("jobs").stream())
    total_tokens = sum(u.to_dict().get("token_balance", 0) for u in users)
    return {
        "total_users":  len(users),
        "total_jobs":   len(jobs),
        "total_tokens": total_tokens,
    }
