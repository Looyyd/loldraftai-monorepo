from enum import Enum
from dataclasses import dataclass
from typing import Callable, Dict, Optional, List
from functools import wraps
import pandas as pd

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def validate_categorical_output(possible_values: List[int]) -> Callable:
    """Decorator to validate that categorical getters only return allowed values."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> pd.Series:
            result = func(*args, **kwargs)
            invalid_values = set(result.unique()) - set(possible_values)
            if invalid_values:
                raise ValueError(
                    f"Getter returned invalid values {invalid_values}. "
                    f"Allowed values are {possible_values}"
                )
            return result

        return wrapper

    return decorator


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
    possible_values: Optional[List[int]] = None

    def __post_init__(self):
        if (
            self.column_type == ColumnType.KNOWN_CATEGORICAL
            and self.possible_values is None
        ):
            raise ValueError(
                f"Column {self.name} is categorical but has no possible_values defined"
            )


possible_values_elo = [0, 1, 2, 3, 4, 5, 6]


@validate_categorical_output(possible_values_elo)
def get_categorical_elo(df: pd.DataFrame) -> pd.Series:
    """Convert tier/division to numerical value."""

    def map_elo(row):
        tier = row["averageTier"]
        division = row["averageDivision"]

        if (tier, division) in [
            ("MASTER", "I"),
            ("GRANDMASTER", "I"),
            ("CHALLENGER", "I"),
        ]:
            return 0
        elif (tier, division) in [
            ("DIAMOND", "II"),
            ("DIAMOND", "I"),
        ]:
            return 1
        elif (tier, division) in [
            ("DIAMOND", "III"),
            ("DIAMOND", "IV"),
        ]:
            return 2
        elif (tier, division) == ("EMERALD", "I"):
            return 3
        elif (tier, division) == ("PLATINUM", "I"):
            return 4
        elif (tier, division) == ("GOLD", "I"):
            return 5
        elif (tier, division) == ("SILVER", "I"):
            return 6
        else:
            raise ValueError(f"Unknown elo: {tier} {division}")

    return df.apply(map_elo, axis=1)


# 0: Ranked Solo/Duo
# 1: Clash
# 2: Reserved for pro play
RANKED_QUEUE_INDEX = 0
CLASH_QUEUE_INDEX = 1
PRO_QUEUE_INDEX = 2

possible_values_queue_type = [
    RANKED_QUEUE_INDEX,
    CLASH_QUEUE_INDEX,
    PRO_QUEUE_INDEX,
]


@validate_categorical_output(possible_values_queue_type)
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

    queue_mapping = {420: RANKED_QUEUE_INDEX, 700: CLASH_QUEUE_INDEX}
    return df["queueId"].map(queue_mapping)


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
        possible_values=possible_values_elo,
    ),
    "queue_type": ColumnDefinition(
        name="queue_type",
        column_type=ColumnType.KNOWN_CATEGORICAL,
        getter=get_categorical_queue_type,
        possible_values=possible_values_queue_type,
    ),
    # special case for patch number, applied in prepare-data.py
    "patch": ColumnDefinition(
        name="patch",
        column_type=ColumnType.SPECIAL,
    ),
    "champion_ids": ColumnDefinition(
        name="champion_ids", column_type=ColumnType.SPECIAL, getter=get_champion_ids
    ),
}

# Helper lists for different column types
KNOWN_CATEGORICAL_COLUMNS_NAMES = [
    col
    for col, def_ in COLUMNS.items()
    if def_.column_type == ColumnType.KNOWN_CATEGORICAL
]
SPECIAL_COLUMNS_NAMES = [
    col for col, def_ in COLUMNS.items() if def_.column_type == ColumnType.SPECIAL
]
