from enum import Enum
from dataclasses import dataclass
from typing import Callable, Dict, Optional
import pandas as pd

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


class ColumnType(Enum):
    # Known categorical, where we know what the values map to
    KNOWN_CATEGORICAL = "known_categorical"
    # special columns, that have unique handling eg:champion ids
    SPECIAL = "special"


@dataclass
class ColumnDefinition:
    name: str
    column_type: ColumnType
    getter: Optional[Callable[[pd.DataFrame], pd.Series]] = None


def get_categorical_elo(df: pd.DataFrame) -> pd.Series:
    """Convert tier/division to numerical value."""

    def map_elo(row):
        tier = row["averageTier"]
        division = row["averageDivision"]

        if (tier, division) in [
            ("DIAMOND", "II"),
            ("DIAMOND", "I"),
            ("MASTER", "I"),
            ("GRANDMASTER", "I"),
            ("CHALLENGER", "I"),
        ]:
            return 0
        elif (tier, division) == ("DIAMOND", "III"):
            return 1
        elif (tier, division) == ("DIAMOND", "IV"):
            return 2
        elif (tier, division) == ("EMERALD", "I"):
            return 3
        else:
            raise ValueError(f"Unknown elo: {tier} {division}")

    return df.apply(map_elo, axis=1)


def get_categorical_queue_type(df: pd.DataFrame) -> pd.Series:
    """Convert queueId to categorical value.
    420: Ranked Solo/Duo -> 0
    700: Clash -> 1
    """
    # Verify the queueId is one of the expected values
    valid_queues = {420, 700}
    invalid_queues = set(df["queueId"].unique()) - valid_queues
    if invalid_queues:
        raise ValueError(f"Unexpected queueId values found: {invalid_queues}")

    queue_mapping = {420: 0, 700: 1}
    return df["queueId"].map(queue_mapping)


# TODO: will need something to know all the possible values for categorical elo!
# TODO: will need something to know all the possible values for categorical queue type!


def get_patch_from_raw_data(df: pd.DataFrame) -> pd.Series:
    """Convert patch version to string format 'major.minor' with zero-padded minor version."""
    return (
        df["gameVersionMajorPatch"].astype(str)
        + "."
        + df["gameVersionMinorPatch"].astype(str).str.zfill(2)
    )


def get_champion_ids(df: pd.DataFrame) -> pd.Series:
    champion_ids = []
    # Blue team, then red team from top to bottom
    for team in [100, 200]:
        for role in POSITIONS:
            champion_ids.append(df[f"team_{team}_{role}_championId"])
    # Concatenate horizontally and apply list conversion to each row
    return pd.concat(champion_ids, axis=1).apply(lambda x: x.tolist(), axis=1)


# Define all columns
COLUMNS: Dict[str, ColumnDefinition] = {
    # Computed columns
    "elo": ColumnDefinition(
        name="elo",
        column_type=ColumnType.KNOWN_CATEGORICAL,
        getter=get_categorical_elo,
    ),
    "queue_type": ColumnDefinition(
        name="queue_type",
        column_type=ColumnType.KNOWN_CATEGORICAL,
        getter=get_categorical_queue_type,
    ),
    # special case for patch number, applied in prepare-data.py
    "patch": ColumnDefinition(
        name="patch",
        column_type=ColumnType.SPECIAL,
        # getter=get_numerical_patch,
    ),
    # TODO: Could change to categorical for simplification
    "champion_ids": ColumnDefinition(
        name="champion_ids", column_type=ColumnType.SPECIAL, getter=get_champion_ids
    ),
}

# Helper lists for different column types
KNOWN_CATEGORICAL_COLUMNS = [
    col
    for col, def_ in COLUMNS.items()
    if def_.column_type == ColumnType.KNOWN_CATEGORICAL
]
SPECIAL_COLUMNS = [
    col for col, def_ in COLUMNS.items() if def_.column_type == ColumnType.SPECIAL
]
