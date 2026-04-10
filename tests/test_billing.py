# tests/test_billing.py
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock
import pytest


# ── Pure function tests (no mocking needed) ────────────────────────────

def test_token_calculation_base_cost_only():
    from app.routers.inference import calculate_tokens
    # 0 sqm area, 0 GB → only base cost
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


# ── Route tests ────────────────────────────────────────────────────────

def test_free_tier_blocked_over_30mb(client, access_token, mocker):
    """Free tier (balance=0) rejects files over 30 MB."""
    mocker.patch(
        "app.routers.inference.asyncio_to_thread_get_user",
        new_callable=AsyncMock,
        return_value={
            "uid": "uid123", "email": "t@t.com", "role": "user",
            "token_balance": 0, "daily_upload_count": 0, "last_upload_date": "",
        },
    )
    big_content = b"0" * (31 * 1024 * 1024)
    resp = client.post(
        "/api/inference",
        files={"file": ("test.tif", big_content, "image/tiff")},
        data={"model_name": "palmCounting-model.onnx"},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert resp.status_code == 403   # our pre-check returns 403 (Forbidden), not 413


def test_free_tier_daily_limit_enforced(client, access_token, mocker):
    """Free tier users who uploaded 3 times today get 403 (pre-check fast-fail)."""
    today = date.today().isoformat()
    mocker.patch(
        "app.routers.inference.asyncio_to_thread_get_user",
        new_callable=AsyncMock,
        return_value={
            "uid": "uid123", "email": "t@t.com", "role": "user",
            "token_balance": 0, "daily_upload_count": 3, "last_upload_date": today,
        },
    )
    resp = client.post(
        "/api/inference",
        files={"file": ("test.tif", b"0" * 1024, "image/tiff")},
        data={"model_name": "palmCounting-model.onnx"},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert resp.status_code == 403   # pre-check (cached count) returns 403, not 429


def test_commercial_tier_insufficient_tokens(client, access_token, mocker):
    """Commercial user with too few tokens gets 402."""
    mocker.patch(
        "app.routers.inference.asyncio_to_thread_get_user",
        new_callable=AsyncMock,
        return_value={
            "uid": "uid123", "email": "t@t.com", "role": "user",
            "token_balance": 10, "daily_upload_count": 0, "last_upload_date": "",
        },
    )
    mocker.patch("app.routers.inference.get_raster_area_sqm", return_value=1_000_000.0)
    # Mock model + YAML existence so we get past the file checks
    mocker.patch.object(Path, "exists", return_value=True)

    resp = client.post(
        "/api/inference",
        files={"file": ("test.tif", b"0" * 1024, "image/tiff")},
        data={"model_name": "palmCounting-model.onnx"},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert resp.status_code == 402


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
