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
        status = 404 if "not found" in str(e).lower() else 400
        raise HTTPException(status, str(e))
    return {"uid": uid, "role": body.role}


@router.get("/stats")
async def admin_stats(admin=Depends(require_role("admin", "superadmin"))):
    """Return global KPI metrics: total users, jobs, and tokens in circulation."""
    stats = await asyncio.to_thread(firestore_client.get_admin_stats)
    return stats


@router.get("/activities")
async def admin_activities(admin=Depends(require_role("admin", "superadmin"))):
    """Return the 200 most recent jobs across all users as an activity feed."""
    jobs = await asyncio.to_thread(firestore_client.get_all_jobs)
    return {"activities": jobs}
