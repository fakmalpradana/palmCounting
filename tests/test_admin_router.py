# tests/test_admin_router.py
from unittest.mock import AsyncMock


def _user(uid, email, role, balance):
    return {
        "uid": uid, "email": email, "name": "Test", "avatar": "",
        "role": role, "token_balance": balance,
        "daily_upload_count": 0, "last_upload_date": "",
    }


def test_list_users_requires_admin(client, access_token):
    resp = client.get("/api/admin/users",
                      headers={"Authorization": f"Bearer {access_token}"})
    assert resp.status_code == 403


def test_list_users_returns_users(client, admin_token, mocker):
    mocker.patch(
        "app.routers.admin.asyncio_to_thread_get_all_users",
        new_callable=AsyncMock,
        return_value=[_user("uid1", "a@b.com", "user", 0)],
    )
    resp = client.get("/api/admin/users",
                      headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    assert len(resp.json()["users"]) == 1


def test_update_tokens_adds_correctly(client, admin_token, mocker):
    mocker.patch(
        "app.routers.admin.asyncio_to_thread_update_tokens",
        new_callable=AsyncMock,
        return_value=150,
    )
    resp = client.patch(
        "/api/admin/users/uid1/tokens",
        json={"delta": 150},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["new_balance"] == 150


def test_update_tokens_requires_admin(client, access_token):
    resp = client.patch(
        "/api/admin/users/uid1/tokens",
        json={"delta": 10},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert resp.status_code == 403


def test_update_role_requires_superadmin(client, admin_token):
    resp = client.patch(
        "/api/admin/users/uid1/role",
        json={"role": "admin"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 403


def test_update_role_by_superadmin(client, superadmin_token, mocker):
    mocker.patch(
        "app.routers.admin.asyncio_to_thread_update_role",
        new_callable=AsyncMock,
        return_value=None,
    )
    resp = client.patch(
        "/api/admin/users/uid1/role",
        json={"role": "admin"},
        headers={"Authorization": f"Bearer {superadmin_token}"},
    )
    assert resp.status_code == 200
