"""
services/refresh_token_service.py — CRUD + rotation logic for RefreshToken.

Mirrors client_service.py's style: plain functions, no classes, the
caller (router) decides what HTTP status maps to what failure.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import generate_refresh_token, hash_token
from app.database.models import Client, RefreshToken

logger = logging.getLogger(__name__)
settings = get_settings()


def issue_refresh_token(db: Session, client_id: str) -> str:
    """
    Create a new refresh token row for this client and return the RAW
    token (the only time the raw value ever exists outside the client's
    hands — only the hash is persisted).
    """
    raw_token = generate_refresh_token()
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)

    token_row = RefreshToken(
        client_id=client_id,
        token_hash=hash_token(raw_token),
        expires_at=expires_at,
    )
    db.add(token_row)
    db.commit()
    return raw_token


def _get_valid_token_row(db: Session, raw_token: str) -> Optional[RefreshToken]:
    """
    Look up a refresh token by its hash and return it only if it's
    neither revoked nor expired. Returns None for any failure case —
    callers must not distinguish "not found" vs "expired" vs "revoked"
    in the response, same principle as authenticate_client().
    """
    token_row = (
        db.query(RefreshToken)
        .filter(RefreshToken.token_hash == hash_token(raw_token))
        .first()
    )
    if not token_row:
        return None
    if token_row.revoked_at is not None:
        return None
    if token_row.expires_at < datetime.now(timezone.utc):
        return None
    return token_row


def rotate_refresh_token(db: Session, raw_token: str) -> Optional[Tuple[str, Client]]:
    """
    Validate an incoming refresh token, revoke it, and issue a new one
    in its place. Returns (new_raw_token, client) on success, None if
    the incoming token was invalid/expired/already revoked/already used.
    """
    token_row = _get_valid_token_row(db, raw_token)
    if not token_row:
        return None

    client = token_row.client
    if not client or not client.is_active:
        return None

    token_row.revoked_at = datetime.now(timezone.utc)
    db.commit()

    new_raw_token = issue_refresh_token(db, client.id)
    return new_raw_token, client


def revoke_refresh_token(db: Session, raw_token: str) -> bool:
    """
    Logout: revoke a single refresh token (one device/session). Returns
    True if a live token was found and revoked, False if it was already
    invalid/expired/revoked — either way the end state is "not usable",
    so the router should treat both as a successful logout.
    """
    token_row = _get_valid_token_row(db, raw_token)
    if not token_row:
        return False
    token_row.revoked_at = datetime.now(timezone.utc)
    db.commit()
    return True