from sqlalchemy import create_engine, Column, String, Integer, Float, TIMESTAMP, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv
from bcrypt import hashpw, gensalt

load_dotenv()

# Use SQLite database file
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///supplyconnect.db")  # Default to local SQLite file if not set
if not DATABASE_URL.startswith("sqlite:///"):
    raise ValueError("DATABASE_URL must be a SQLite URL (e.g., sqlite:///supplyconnect.db). Check your .env file.")

# SQLite-specific engine configuration
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})  # Required for SQLite in multi-threaded apps
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dealer Table
class Dealer(Base):
    __tablename__ = "dealers"
    
    dealer_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    company_name = Column(String(100), nullable=False)
    location_name = Column(String(100), nullable=False)
    latitude = Column(Float)  # Changed from DECIMAL to Float
    longitude = Column(Float)  # Changed from DECIMAL to Float
    password_hash = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

# Shopkeeper Table
class Shopkeeper(Base):
    __tablename__ = "shopkeepers"
    
    shopkeeper_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    shop_name = Column(String(100), nullable=False)
    location_name = Column(String(100), nullable=False)
    latitude = Column(Float)  # Changed from DECIMAL to Float
    longitude = Column(Float)  # Changed from DECIMAL to Float
    dealer_id = Column(Integer, ForeignKey("dealers.dealer_id"), nullable=True)
    domain = Column(String(50))
    password_hash = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

# Create all tables
Base.metadata.create_all(bind=engine)