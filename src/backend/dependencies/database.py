import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.backend.models import Base
from src.backend.core.init_settings import global_settings as settings

# Synchronous engine and session
sync_engine = create_engine(settings.DB_URL)
SyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

def get_db():
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    # Check if tables are already created (assuming there's at least one table)
    if not os.path.exists("./dev.db"):
        Base.metadata.create_all(bind=sync_engine)
        print("Database initiate successfully!")