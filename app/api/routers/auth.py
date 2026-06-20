"""api/routers/auth.py — Signup and login."""
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import create_access_token
from app.database.db import get_db
from app.schemas.auth import SignupRequest, LoginRequest, TokenResponse
from app.services.client_service import create_client_with_password, authenticate_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup(request: SignupRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """
    Create a new account (or attach credentials to a Client row that was
    created anonymously via chat's client_email field, if one exists and
    has no password yet). Returns a token immediately — no email
    verification step in this version.
    """
    try:
        client = create_client_with_password(db, request.email, request.password, request.name)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

    token = create_access_token(subject=client.id)
    return TokenResponse(access_token=token, client_id=client.id, email=client.email, name=client.name)


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Verify email/password and return a JWT. Generic 401 on any failure — never reveals which part was wrong."""
    client = authenticate_client(db, request.email, request.password)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(subject=client.id)
    return TokenResponse(access_token=token, client_id=client.id, email=client.email, name=client.name)