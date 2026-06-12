# test_db.py
from app.core.config import get_settings
from app.database.db import engine, init_db
from sqlalchemy import text

settings = get_settings()
print(f"Connecting to: {settings.database_url}")

# Test connection
with engine.connect() as conn:
    result = conn.execute(text("SELECT version()"))
    print(f"PostgreSQL version: {result.scalar()}")

# Create tables
init_db()
print("Tables created successfully!")