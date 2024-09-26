# database.py
import os
import enum

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    DateTime,
    Boolean,
    JSON,
    Enum,
    UniqueConstraint,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func


DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set")

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class Region(enum.Enum):
    BR1 = "BR1"
    EUN1 = "EUN1"
    EUW1 = "EUW1"
    JP1 = "JP1"
    KR = "KR"
    LA1 = "LA1"
    LA2 = "LA2"
    ME1 = "ME1"
    NA1 = "NA1"
    OC1 = "OC1"
    PH2 = "PH2"
    RU = "RU"
    SG2 = "SG2"
    TH2 = "TH2"
    TR1 = "TR1"
    TW2 = "TW2"
    VN2 = "VN2"


class Tier(enum.Enum):
    CHALLENGER = "CHALLENGER"
    GRANDMASTER = "GRANDMASTER"
    MASTER = "MASTER"
    DIAMOND = "DIAMOND"
    EMERALD = "EMERALD"
    PLATINUM = "PLATINUM"
    GOLD = "GOLD"
    SILVER = "SILVER"
    BRONZE = "BRONZE"
    IRON = "IRON"


class Division(enum.Enum):
    I = "I"
    II = "II"
    III = "III"
    IV = "IV"


class Match(Base):
    __tablename__ = "Match"

    id = Column(String, primary_key=True, index=True)
    matchId = Column(String, nullable=False)
    queueId = Column(Integer, nullable=True)
    region = Column(Enum(Region), index=True, nullable=False)
    averageTier = Column(Enum(Tier), nullable=False)
    averageDivision = Column(Enum(Division), nullable=False)
    gameVersionMajorPatch = Column(Integer, nullable=True)
    gameVersionMinorPatch = Column(Integer, nullable=True)
    gameDuration = Column(Integer, nullable=True)
    gameStartTimestamp = Column(DateTime, nullable=True)
    processed = Column(Boolean, default=False, index=True)
    processingErrored = Column(Boolean, default=False, index=True)
    teams = Column(JSON, nullable=True)
    createdAt = Column(DateTime, server_default=func.now())
    updatedAt = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("matchId", "region", name="uq_match_id_region"),
        Index("idx_processed_errored", "processed", "processingErrored"),
        Index("idx_game_start_timestamp", "gameStartTimestamp"),
    )


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
