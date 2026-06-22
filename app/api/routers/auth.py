"""api/routers/auth.py — Signup, login, and the authenticated client's own data."""
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_client
from app.core.security import create_access_token
from app.database.db import get_db
from app.database.models import Client
from app.schemas.auth import (
    SignupRequest, LoginRequest, TokenResponse, ClientProfileResponse,
    RefreshRequest, LogoutRequest,
)
from app.schemas.client import BusinessPlanSummary, KPISnapshotOut
from app.services.client_service import create_client_with_password, authenticate_client
from app.services.refresh_token_service import issue_refresh_token, rotate_refresh_token, revoke_refresh_token
from app.services.plan_service import get_plans_by_client
from app.services.kpi_service import get_kpis_by_client

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
    refresh_token = issue_refresh_token(db, client.id)
    return TokenResponse(
        access_token=token, refresh_token=refresh_token,
        client_id=client.id, email=client.email, name=client.name,
    )


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
    refresh_token = issue_refresh_token(db, client.id)
    return TokenResponse(
        access_token=token, refresh_token=refresh_token,
        client_id=client.id, email=client.email, name=client.name,
    )


@router.get("/me", response_model=ClientProfileResponse)
def read_current_client(current_client: Client = Depends(get_current_client)) -> Client:
    """
    Return the authenticated client's own profile. Requires a valid bearer
    token — unlike /chat's optional auth, there's no "anonymous" version of
    this endpoint, since there's nothing to return without an identity.
    """
    return current_client


@router.get("/me/plans", response_model=List[BusinessPlanSummary])
def list_my_plans(
    current_client: Client = Depends(get_current_client),
    db: Session = Depends(get_db),
) -> List[BusinessPlanSummary]:
    """
    List business plans generated across all of this client's sessions,
    most recent first. Summary fields only — fetch a specific plan by id
    (future endpoint) for the full P&L/cash-flow/financing detail.
    """
    return get_plans_by_client(db, current_client.id)


@router.get("/me/kpis", response_model=List[KPISnapshotOut])
def list_my_kpis(
    current_client: Client = Depends(get_current_client),
    db: Session = Depends(get_db),
) -> List[KPISnapshotOut]:
    """List every KPI snapshot ever calculated for this client, most recent first."""
    return get_kpis_by_client(db, current_client.id)


@router.post("/refresh", response_model=TokenResponse)
def refresh(request: RefreshRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """
    Exchange a still-valid refresh token for a brand new access token AND
    a brand new refresh token (rotation — the one sent in this request is
    immediately revoked, so it can't be replayed). Generic 401 on any
    failure, same principle as /login: don't reveal whether the token was
    invalid, expired, or already used.
    """
    result = rotate_refresh_token(db, request.refresh_token)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    new_refresh_token, client = result
    new_access_token = create_access_token(subject=client.id)
    return TokenResponse(
        access_token=new_access_token, refresh_token=new_refresh_token,
        client_id=client.id, email=client.email, name=client.name,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: LogoutRequest, db: Session = Depends(get_db)) -> None:
    """
    Revoke a refresh token (one device/session logged out). No access
    token required — whoever holds a valid refresh token could mint a
    fresh access token anyway via /refresh, so skipping the access-token
    check here doesn't weaken anything. Always returns 204, even if the
    token was already invalid/expired — logout is idempotent from the
    caller's point of view.
    """
    revoke_refresh_token(db, request.refresh_token)
    return None