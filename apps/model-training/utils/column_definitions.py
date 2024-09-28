# utils/column_definitions.py
from enum import Enum
from typing import Any, Callable, Dict

from utils.database import Match


class ColumnType(Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    LIST = "list"


# Type alias for the extraction function
ExtractorFunc = Callable[[Match], Any]


class ColumnDefinition:
    def __init__(self, column_type: ColumnType, extractor: ExtractorFunc):
        self.column_type = column_type
        self.extractor = extractor


def extract_region(match: Match):
    return match.region.value


def extract_average_tier(match: Match):
    return match.averageTier.value


def extract_average_division(match: Match):
    return match.averageDivision.value


def extract_game_version_major_patch(match: Match):
    # TODO: should always be set, maybe remove -1?
    return (
        match.gameVersionMajorPatch if match.gameVersionMajorPatch is not None else -1
    )


def extract_game_version_minor_patch(match: Match):
    return (
        match.gameVersionMinorPatch if match.gameVersionMinorPatch is not None else -1
    )


POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def extract_champion_ids(match: Match):
    champion_ids = []
    for team_id in sorted(match.teams.keys()):
        team = match.teams[team_id]
        for position in POSITIONS:
            champion_ids.append(team["participants"][position]["championId"])
    return champion_ids


COLUMNS: Dict[str, ColumnDefinition] = {
    "region": ColumnDefinition(ColumnType.CATEGORICAL, extract_region),
    "averageTier": ColumnDefinition(ColumnType.CATEGORICAL, extract_average_tier),
    "averageDivision": ColumnDefinition(
        ColumnType.CATEGORICAL, extract_average_division
    ),
    "champion_ids": ColumnDefinition(ColumnType.LIST, extract_champion_ids),
    "gameVersionMajorPatch": ColumnDefinition(
        ColumnType.NUMERICAL, extract_game_version_major_patch
    ),
    "gameVersionMinorPatch": ColumnDefinition(
        ColumnType.NUMERICAL, extract_game_version_minor_patch
    ),
}

CATEGORICAL_COLUMNS = [
    col for col, def_ in COLUMNS.items() if def_.column_type == ColumnType.CATEGORICAL
]
NUMERICAL_COLUMNS = [
    col for col, def_ in COLUMNS.items() if def_.column_type == ColumnType.NUMERICAL
]
LIST_COLUMNS = [
    col for col, def_ in COLUMNS.items() if def_.column_type == ColumnType.LIST
]


def extract_raw_features(match: Match):
    return {col: def_.extractor(match) for col, def_ in COLUMNS.items()}
