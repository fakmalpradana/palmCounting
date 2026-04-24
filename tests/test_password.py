# tests/test_password.py
from app.core.password import hash_password, verify_password


def test_hash_and_verify_correct():
    hashed = hash_password("mysecretpw")
    assert verify_password("mysecretpw", hashed)


def test_verify_wrong_password():
    hashed = hash_password("correct-password")
    assert not verify_password("wrong-password", hashed)


def test_hash_produces_different_hashes():
    h1 = hash_password("same-password")
    h2 = hash_password("same-password")
    assert h1 != h2  # bcrypt salts should differ
