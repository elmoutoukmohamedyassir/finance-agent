"""
api/deps.py — FastAPI auth dependencies.

Two flavors, used depending on the route's needs:

  - get_current_client: 401s if there's no valid token. Use on routes that
    must have an identified owner (none yet in v1, but e.g. a future
    "my plans" listing endpoint would use this).

  - get_current_client_optional: returns the Client if a valid token is
    present, else None — never raises for a missing/invalid token. This is
    what /chat uses, since anonymous chat must keep working (decision: chat
    stays open, only attaches an owner when the caller is logged in).

Both use HTTPBearer (not OAuth2PasswordBearer) so Swagger's "Authorize"
button takes a raw bearer token rather than expecting an OAuth2 password
flow form — matches the established pattern from the auth branch.
"""
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.database.db import get_db
from app.database.models import Client
from app.services.client_service import get_client_by_id

# auto_error=False so a missing header doesn't 403 by itself — we decide
# what "missing" means per-dependency below (required vs optional).
_bearer_scheme = HTTPBearer(auto_error=False)


def _resolve_client(
    credentials: Optional[HTTPAuthorizationCredentials], db: Session
) -> Optional[Client]:
    if credentials is None:
        return None
    client_id = decode_access_token(credentials.credentials)
    if not client_id:
        return None
    client = get_client_by_id(db, client_id)
    if not client or not client.is_active:
        return None
    return client


def get_current_client(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> Client:
    """Require a valid bearer token. 401 if missing, malformed, expired, or for an inactive account."""
    client = _resolve_client(credentials, db)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return client


def get_current_client_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> Optional[Client]:
    """Best-effort auth: returns the Client if the token is valid, otherwise None. Never raises."""
    return _resolve_client(credentials, db)