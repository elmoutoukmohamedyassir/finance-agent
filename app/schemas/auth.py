"""schemas/auth.py — Request/response models for signup and login."""
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=72, description="8-72 characters (bcrypt limit)")
    name: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "email": "entrepreneur@startup.ma",
                "password": "a-strong-passphrase",
                "name": "Yassine El Amrani",
            }
        }
    }


class LoginRequest(BaseModel):
    email: EmailStr
    password: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "email": "entrepreneur@startup.ma",
                "password": "a-strong-passphrase",
            }
        }
    }


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    client_id: str
    email: Optional[str] = None
    name: Optional[str] = None