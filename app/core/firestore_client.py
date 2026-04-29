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

# In-memory stores for dev mode (email/password users + email index).
_DEV_USERS: dict[str, dict] = {}   # uid → user dict
_DEV_EMAIL_INDEX: dict[str, str] = {}  # email → uid


def get_db():
    global _db
    if _db is None:
        from google.cloud import firestore
        _db = firestore.Client(project=settings.firestore_project_id)
    return _db


def set_email_index(email: str, user_uid: str) -> None:
    """Create or update the email → user_uid lookup in the user_emails collection."""
    if settings.dev_mode:
        _DEV_EMAIL_INDEX[email.lower()] = user_uid
        return
    db = get_db()
    db.collection("user_emails").document(email.lower()).set({"user_uid": user_uid})


def get_user_by_email(email: str) -> Optional[dict]:
    """Look up a user by email via the user_emails index collection."""
    if settings.dev_mode:
        uid = _DEV_EMAIL_INDEX.get(email.lower())
        if uid:
            return _DEV_USERS.get(uid)
        if email.lower() == _DEV_USER["email"]:
            return _DEV_USER
        return None
    db = get_db()
    index_doc = db.collection("user_emails").document(email.lower()).get()
    if not index_doc.exists:
        return None
    user_uid = index_doc.to_dict().get("user_uid")
    if not user_uid:
        return None
    return get_user(user_uid)


def create_email_user(email: str, name: str, password_hash: str) -> dict:
    """Register a new user with email/password. Returns the user dict."""
    if settings.dev_mode:
        uid = f"ep_{uuid.uuid4()}"
        user = {
            **{k: v for k, v in _DEV_USER.items() if k != "uid"},
            "uid": uid, "email": email.lower(), "name": name,
            "role": "superadmin" if email == SUPERADMIN_EMAIL else "user",
            "token_balance": 0, "password_hash": password_hash,
            "auth_provider": "email", "email_verified": False,
        }
        _DEV_USERS[uid] = user
        _DEV_EMAIL_INDEX[email.lower()] = uid
        return user
    uid = f"ep_{uuid.uuid4()}"
    role = "superadmin" if email == SUPERADMIN_EMAIL else "user"
    data = {
        "email":               email.lower(),
        "name":                name,
        "avatar":              "",
        "role":                role,
        "token_balance":       0,
        "daily_upload_count":  0,
        "last_upload_date":    "",
        "free_palm_used":      False,
        "free_landcover_used": False,
        "password_hash":       password_hash,
        "auth_provider":       "email",
        "email_verified":      False,
    }
    db = get_db()
    db.collection("users").document(uid).set(data)
    set_email_index(email, uid)
    return {**data, "uid": uid}


def upsert_user(google_uid: str, email: str, name: str, avatar: str) -> dict:
    if settings.dev_mode:
        return _DEV_USER
    db = get_db()

    # Check if an email/password user already exists with this email (account linking).
    index_doc = db.collection("user_emails").document(email.lower()).get()
    if index_doc.exists:
        existing_uid = index_doc.to_dict().get("user_uid", "")
        if existing_uid and existing_uid != google_uid:
            # An email/password account exists — link Google to it.
            ref = db.collection("users").document(existing_uid)
            doc = ref.get()
            if doc.exists:
                existing = doc.to_dict()
                role = "superadmin" if email == SUPERADMIN_EMAIL else existing.get("role", "user")
                updates = {
                    "name": name,
                    "avatar": avatar,
                    "role": role,
                    "auth_provider": "both",
                    "google_uid": google_uid,
                    "email_verified": True,
                }
                ref.update(updates)
                return {**existing, **updates, "uid": existing_uid}

    # Standard Google OAuth path (no pre-existing email/password account).
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
        if "auth_provider" not in existing:
            updates["auth_provider"] = "google"
        if "email_verified" not in existing:
            updates["email_verified"] = True
        ref.update(updates)
        set_email_index(email, google_uid)
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
            "auth_provider":       "google",
            "email_verified":      True,
        }
        ref.set(data)
        set_email_index(email, google_uid)
        return {**data, "uid": google_uid}


def get_user(google_uid: str) -> Optional[dict]:
    if settings.dev_mode:
        return _DEV_USERS.get(google_uid, _DEV_USER)
    db = get_db()
    doc = db.collection("users").document(google_uid).get()
    if not doc.exists:
        return None
    return {**doc.to_dict(), "uid": google_uid}


def update_user_password(user_uid: str, password_hash: str) -> None:
    """Set or update the password hash on a user doc (for registration or reset)."""
    if settings.dev_mode:
        user = _DEV_USERS.get(user_uid)
        if user:
            user["password_hash"] = password_hash
            if user.get("auth_provider") == "google":
                user["auth_provider"] = "both"
        return
    db = get_db()
    ref = db.collection("users").document(user_uid)
    updates = {"password_hash": password_hash}
    doc = ref.get()
    if doc.exists:
        existing = doc.to_dict()
        if existing.get("auth_provider") == "google":
            updates["auth_provider"] = "both"
    ref.update(updates)


def set_email_verified(user_uid: str) -> None:
    """Mark a user's email as verified."""
    if settings.dev_mode:
        user = _DEV_USERS.get(user_uid)
        if user:
            user["email_verified"] = True
        return
    db = get_db()
    db.collection("users").document(user_uid).update({"email_verified": True})


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
