from datetime import datetime, timedelta, timezone

import jwt
import pytest

from app.common.auth import (
    UnauthorizedError,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user_id,
    get_optional_user_id,
    hash_password,
    verify_password,
)
from app.core.config import settings


def test_argon2_hash_salted_and_verify_positive():
    h1 = hash_password("Secret@2026")
    h2 = hash_password("Secret@2026")
    assert h1 != "Secret@2026"
    assert h1 != h2
    assert verify_password("Secret@2026", h1) is True
    assert verify_password("Secret@2026", h2) is True


@pytest.mark.parametrize(
    ("plain", "hashed"),
    [
        ("wrong-password", hash_password("Secret@2026")),
        ("", hash_password("Secret@2026")),
        ("Secret@2026", ""),
        ("123456", "$2a$12$demo.password.hash.placeholder"),
        ("123456", "not-a-valid-argon2-hash"),
    ],
)
def test_verify_password_rejects_invalid_or_corrupt_hashes(plain: str, hashed: str):
    assert verify_password(plain, hashed) is False


def test_access_and_refresh_token_roundtrip():
    access = create_access_token(42)
    refresh = create_refresh_token(42)
    assert decode_token(access, expected_type="access") == 42
    assert decode_token(refresh, expected_type="refresh") == 42


def test_decode_token_rejects_type_confusion():
    access = create_access_token(7)
    refresh = create_refresh_token(7)

    with pytest.raises(UnauthorizedError, match="Invalid token type"):
        decode_token(refresh, expected_type="access")
    with pytest.raises(UnauthorizedError, match="Invalid token type"):
        decode_token(access, expected_type="refresh")


def test_decode_token_rejects_expired_token():
    expired = jwt.encode(
        {
            "sub": "9",
            "type": "access",
            "iat": datetime.now(timezone.utc) - timedelta(hours=2),
            "exp": datetime.now(timezone.utc) - timedelta(seconds=5),
        },
        settings.app.jwt_secret_key,
        algorithm="HS256",
    )
    with pytest.raises(UnauthorizedError, match="Invalid or expired token"):
        decode_token(expired, expected_type="access")


def test_decode_token_rejects_wrong_secret_and_non_digit_subject():
    forged = jwt.encode(
        {
            "sub": "9",
            "type": "access",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
        },
        "completely-wrong-secret-key-for-test-only",
        algorithm="HS256",
    )
    with pytest.raises(UnauthorizedError):
        decode_token(forged, expected_type="access")

    bad_sub = jwt.encode(
        {
            "sub": "admin",
            "type": "access",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
        },
        settings.app.jwt_secret_key,
        algorithm="HS256",
    )
    with pytest.raises(UnauthorizedError, match="Invalid token subject"):
        decode_token(bad_sub, expected_type="access")


def test_get_current_user_id_accepts_bearer_and_prefers_over_x_user_id():
    token = create_access_token(15)
    assert get_current_user_id(authorization=f"Bearer {token}", x_user_id=None) == 15
    assert get_current_user_id(authorization=f"bearer  {token} ", x_user_id=99) == 15


@pytest.mark.parametrize(
    ("authorization", "x_user_id"),
    [
        (None, None),
        (None, 0),
        (None, -3),
        ("Basic dXNlcjpwYXNz", 1),
        ("Bearer ", 1),
        ("Bearer not.a.jwt", None),
        (f"Bearer {create_refresh_token(1)}", None),
    ],
)
def test_get_current_user_id_rejects_missing_or_invalid_credentials(
    authorization: str | None,
    x_user_id: int | None,
):
    with pytest.raises(UnauthorizedError):
        get_current_user_id(authorization=authorization, x_user_id=x_user_id)


def test_get_current_user_id_allows_positive_x_user_id_fallback():
    assert get_current_user_id(authorization=None, x_user_id=3) == 3


def test_get_optional_user_id_degrades_safely_on_invalid_or_missing_headers():
    valid_access = create_access_token(88)
    valid_refresh = create_refresh_token(88)

    assert get_optional_user_id(authorization=f"Bearer {valid_access}", x_user_id=None) == 88
    assert get_optional_user_id(authorization=None, x_user_id=6) == 6
    assert get_optional_user_id(authorization=None, x_user_id=None) is None
    assert get_optional_user_id(authorization=None, x_user_id=0) is None
    assert get_optional_user_id(authorization="Basic abc", x_user_id=6) is None
    assert get_optional_user_id(authorization=f"Bearer {valid_refresh}", x_user_id=6) is None
    assert get_optional_user_id(authorization="Bearer invalid.jwt.token", x_user_id=None) is None
