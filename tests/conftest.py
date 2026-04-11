# tests/conftest.py
import os

# Set env vars BEFORE any app imports — required fields have no defaults.
# Use os.environ[] (not setdefault) for DEV_MODE so .env.local cannot override
# the test environment and cause Firestore mocks to be bypassed.
os.environ["DEV_MODE"]               = "false"
os.environ.setdefault("GOOGLE_CLIENT_ID",      "test-client-id")
os.environ.setdefault("GOOGLE_CLIENT_SECRET",  "test-client-secret")
os.environ.setdefault("JWT_SECRET",            "test-jwt-secret-32-bytes-long-ok!")
os.environ.setdefault("FIRESTORE_PROJECT_ID",  "test-project")
os.environ.setdefault("FRONTEND_URL",          "http://localhost:8080")

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def mock_firestore_db(mocker):
    """Patch the Firestore lazy getter so no real DB connection is made."""
    mock_db = MagicMock()
    mocker.patch("app.core.firestore_client.get_db", return_value=mock_db)
    return mock_db


@pytest.fixture
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def access_token():
    from app.core.auth import create_access_token
    return create_access_token({"sub": "uid123", "email": "test@example.com", "role": "user"})


@pytest.fixture
def admin_token():
    from app.core.auth import create_access_token
    return create_access_token({"sub": "uid_admin", "email": "admin@example.com", "role": "admin"})


@pytest.fixture
def superadmin_token():
    from app.core.auth import create_access_token
    return create_access_token({"sub": "uid_super", "email": "fakmalpradana@gmail.com", "role": "superadmin"})
