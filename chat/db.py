# db.py
from sqlalchemy import create_engine, Column, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database URL (using SQLite)
DATABASE_URL = "sqlite:///./app.db"

# Create engine and session
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()

# User Profile model
class UserProfileDB(Base):
    __tablename__ = "user_profiles"
    user_id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    attributes = Column(JSON, nullable=False)

# Chat History model (optional for persisting chat)
class ChatHistoryDB(Base):
    __tablename__ = "chat_histories"
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    messages = Column(JSON, nullable=False)

# Create tables
Base.metadata.create_all(bind=engine)

