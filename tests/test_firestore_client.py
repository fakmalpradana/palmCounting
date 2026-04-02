# tests/test_firestore_client.py
from unittest.mock import MagicMock
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
