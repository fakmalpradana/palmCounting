# app/core/firestore_client.py
from __future__ import annotations

from datetime import date
from typing import Optional

from app.core.config import settings

SUPERADMIN_EMAIL = "fakmalpradana@gmail.com"

# ── Dev-mode mock user (used when DEV_MODE=true, Firestore is never contacted) ──
_DEV_USER: dict = {
    "uid":               "dev-local-user",
    "email":             "dev@local.test",
    "name":              "Dev User",
    "avatar":            "",
    "role":              "superadmin",
    "token_balance":     9999,
    "daily_upload_count": 0,
    "last_upload_date":  "",
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
