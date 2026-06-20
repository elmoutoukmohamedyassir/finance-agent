"""
core/security.py — Password hashing and JWT encode/decode helpers.

All auth-token logic goes through here, same pattern as groq_client.py:
one place owns the external library calls, everything else just imports
plain functions.

Password hashing: passlib[bcrypt]. bcrypt has a hard 72-byte input limit —
passlib handles this internally (raises rather than silently truncating on
modern passlib versions), so we don't need to do anything special for it.

JWT: python-jose. Tokens carry the Client's id in `sub` (subject) — the
standard JWT claim for "who this token is about" — plus an expiry (`exp`).
We deliberately keep the payload minimal: no email, no role, nothing that
could go stale and disagree with the DB. The token is only ever used to
look up the Client by id; everything else is read fresh from Postgres on
every request.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import jwt, JWTError
from passlib.context import CryptContext

from app.core.config import get_settings

settings = get_settings()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password for storage. Never store plain_password itself."""
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check a plaintext password against a stored hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str, expires_minutes: Optional[int] = None) -> str:
    """
    Create a signed JWT for the given subject (Client.id as a string).

    expires_minutes overrides settings.access_token_expire_minutes — mainly
    useful for tests; production code should rely on the default.
    """
    if not settings.secret_key:
        raise RuntimeError(
            "SECRET_KEY is not set. Add SECRET_KEY=<a long random string> to your .env. "
            "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
        )

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes if expires_minutes is not None else settings.access_token_expire_minutes
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> Optional[str]:
    """
    Decode a JWT and return the subject (Client.id) if valid, else None.
    Never raises — callers treat None as "not authenticated" and decide
    whether that's an error (required auth) or fine (optional auth).
    """
    if not settings.secret_key:
        raise RuntimeError(
            "SECRET_KEY is not set. Add SECRET_KEY=<a long random string> to your .env."
        )

    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        return payload.get("sub")
    except JWTError:
        return None