"""
services/client_service.py — CRUD operations for Client model.

Client doubles as the auth identity (see database/models.py). The
get_or_create_client() function below is the original, password-agnostic
helper used by the anonymous chat flow (ChatRequest.client_email) — it
must keep working exactly as before for clients who never sign up.
create_client_with_password() and authenticate_client() are the new
auth-specific entry points used by api/routers/auth.py.
"""
import uuid
import logging
from typing import Optional
from sqlalchemy.orm import Session

from app.database.models import Client
from app.core.security import hash_password, verify_password

logger = logging.getLogger(__name__)


def get_or_create_client(
    db: Session, 
    email: Optional[str] = None, 
    name: Optional[str] = None,
    phone: Optional[str] = None,
    sector: Optional[str] = None
) -> Client:
    """Get existing client by email or create new one."""
    if email:
        client = db.query(Client).filter(Client.email == email).first()
        if client:
            return client
    
    client = Client(
        id=str(uuid.uuid4()),
        email=email,
        name=name,
        phone=phone,
        sector=sector
    )
    db.add(client)
    db.commit()
    db.refresh(client)
    logger.info(f"Created new client: {client.id}")
    return client


def get_client_by_id(db: Session, client_id: str) -> Optional[Client]:
    """Retrieve client by ID."""
    return db.query(Client).filter(Client.id == client_id).first()


def get_client_by_email(db: Session, email: str) -> Optional[Client]:
    """Retrieve client by email."""
    return db.query(Client).filter(Client.email == email).first()


def update_client(db: Session, client_id: str, **kwargs) -> Optional[Client]:
    """Update client fields."""
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        return None
    
    for key, value in kwargs.items():
        if hasattr(client, key) and value is not None:
            setattr(client, key, value)
    
    db.commit()
    db.refresh(client)
    logger.info(f"Updated client: {client_id}")
    return client


def create_client_with_password(
    db: Session, email: str, password: str, name: Optional[str] = None
) -> Client:
    """
    Sign up: create a new Client with a hashed password, OR — if a Client
    with that email already exists (e.g. created anonymously via chat's
    client_email field) and has no password yet — "claim" it by setting
    the password on the existing row instead of erroring. This matches
    how get_or_create_client() already treats email as the de-facto unique
    identity; signup just adds credentials to it.

    Raises ValueError if the email is already a fully registered account
    (has a password already) — caller maps this to HTTP 409.
    """
    existing = get_client_by_email(db, email)
    if existing:
        if existing.hashed_password:
            raise ValueError("An account with this email already exists.")
        existing.hashed_password = hash_password(password)
        if name and not existing.name:
            existing.name = name
        db.commit()
        db.refresh(existing)
        logger.info(f"Claimed existing client with password: {existing.id}")
        return existing

    client = Client(
        id=str(uuid.uuid4()),
        email=email,
        name=name,
        hashed_password=hash_password(password),
    )
    db.add(client)
    db.commit()
    db.refresh(client)
    logger.info(f"Signed up new client: {client.id}")
    return client


def authenticate_client(db: Session, email: str, password: str) -> Optional[Client]:
    """
    Verify email/password. Returns the Client on success, None on any
    failure (wrong email, wrong password, no password set, inactive
    account) — callers must not distinguish these cases in the response,
    to avoid leaking which emails are registered.
    """
    client = get_client_by_email(db, email)
    if not client or not client.hashed_password or not client.is_active:
        return None
    if not verify_password(password, client.hashed_password):
        return None
    return client