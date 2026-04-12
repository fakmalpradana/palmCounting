# app/routers/jobs.py
"""
Job Management API
------------------
GET  /api/jobs               — List all jobs for the authenticated user
POST /api/jobs               — Create a new job
PATCH /api/jobs/{job_id}     — Update job name or status
"""
from __future__ import annotations

import asyncio
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core import firestore_client
from app.middleware.auth import get_current_user

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


# ── Pydantic models ────────────────────────────────────────────────────────────

class CreateJobRequest(BaseModel):
    job_name: str = Field(default="Untitled Job", max_length=128)
    task_type: Literal["palm_counting", "land_cover"]
    parameters: dict[str, Any] = Field(default_factory=dict)
    file_uri: str = Field(default="")


class PatchJobRequest(BaseModel):
    job_name: Optional[str] = Field(default=None, max_length=128)
    status: Optional[Literal["uploading", "approval", "processing", "done", "failed"]] = None
    file_uri: Optional[str] = None
    result_uri: Optional[str] = None
    token_cost: Optional[int] = None
    parameters: Optional[dict[str, Any]] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _run(fn, *args, **kwargs):
    """Run a blocking Firestore call in a thread pool."""
    return await asyncio.to_thread(fn, *args, **kwargs)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("")
async def list_jobs(user: dict = Depends(get_current_user)):
    """Return all jobs owned by the authenticated user, newest first."""
    jobs = await _run(firestore_client.get_jobs_for_user, user["email"])
    return {"jobs": jobs}


@router.post("", status_code=201)
async def create_job(
    body: CreateJobRequest,
    user: dict = Depends(get_current_user),
):
    """Create a new job record. Status starts as 'uploading'."""
    job = await _run(
        firestore_client.create_job,
        user["email"],
        body.job_name,
        body.task_type,
        body.parameters,
        body.file_uri,
    )
    return job


@router.patch("/{job_id}")
async def patch_job(
    job_id: str,
    body: PatchJobRequest,
    user: dict = Depends(get_current_user),
):
    """Update job_name and/or status (and optionally other fields).
    Only the owning user (or an admin) may modify a job.
    """
    # Fetch the job first to verify ownership
    existing = await _run(firestore_client.get_job, job_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Job not found")

    is_admin = user.get("role") in ("admin", "superadmin")
    if existing["user_email"] != user["email"] and not is_admin:
        raise HTTPException(status_code=403, detail="Not authorised to modify this job")

    # Build update dict from non-None fields only
    updates: dict[str, Any] = {
        k: v
        for k, v in body.model_dump().items()
        if v is not None
    }
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    updated = await _run(firestore_client.update_job, job_id, updates)
    return updated
