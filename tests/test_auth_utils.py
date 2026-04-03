# tests/test_auth_utils.py
from datetime import timedelta
import pytest


def test_access_token_contains_correct_claims():
    from app.core.auth import create_access_token, decode_token
    token = create_access_token({"sub": "uid1", "email": "a@b.com", "role": "user"})
    payload = decode_token(token)
    assert payload["sub"] == "uid1"
    assert payload["email"] == "a@b.com"
    assert payload["role"] == "user"
    assert payload["type"] == "access"


def test_refresh_token_contains_correct_claims():
    from app.core.auth import create_refresh_token, decode_token
    token = create_refresh_token({"sub": "uid1"})
    payload = decode_token(token)
    assert payload["sub"] == "uid1"
    assert payload["type"] == "refresh"


def test_decode_expired_token_raises():
    from app.core.auth import create_access_token, decode_token
    from jose import ExpiredSignatureError
    token = create_access_token({"sub": "uid1"}, expires_delta=timedelta(seconds=-1))
    with pytest.raises(ExpiredSignatureError):
        decode_token(token)


def test_decode_invalid_token_raises():
    from app.core.auth import decode_token
    from jose import JWTError
    with pytest.raises(JWTError):
        decode_token("not.a.valid.token")
