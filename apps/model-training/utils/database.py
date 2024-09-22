# database.py
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os


DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set")

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# Define the Match model
class Match(Base):
    __tablename__ = "Match"

    id = Column(String, primary_key=True, index=True)
    matchId = Column(String, unique=True, index=True, nullable=False)
    queueId = Column(Integer, nullable=True)
    region = Column(String, index=True, nullable=False)
    averageTier = Column(String, nullable=False)
    averageDivision = Column(String, nullable=False)
    gameVersionMajorPatch = Column(Integer, nullable=True)
    gameVersionMinorPatch = Column(Integer, nullable=True)
    gameDuration = Column(Integer, nullable=True)
    gameStartTimestamp = Column(DateTime, nullable=True)
    processed = Column(Boolean, default=False, index=True)
    teams = Column(JSON, nullable=True)


# Function to create a new session
def get_session():
    return SessionLocal()


# Function to fetch data in batches using ID ranges
def fetch_matches_batch(session, last_id, batch_size=1000, region=None):
    query = session.query(Match).filter(Match.processed == True)
    if region:
        query = query.filter(Match.region == region)
    if last_id:
        query = query.filter(Match.id > last_id)
    query = query.order_by(Match.id).limit(batch_size)
    return query.all()
