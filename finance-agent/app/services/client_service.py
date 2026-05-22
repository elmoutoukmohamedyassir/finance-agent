"""
services/client_service.py — CRUD operations for Client model.
"""
import uuid
import logging
from typing import Optional
from sqlalchemy.orm import Session

from app.database.models import Client

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
