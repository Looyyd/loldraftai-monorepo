from enum import Enum


class ColumnType(Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    LIST = "list"


COLUMNS = {
    "region": ColumnType.CATEGORICAL,
    "averageTier": ColumnType.CATEGORICAL,
    "averageDivision": ColumnType.CATEGORICAL,
    "champion_ids": ColumnType.LIST,
    # Add more columns as needed
}

CATEGORICAL_COLUMNS = [
    col for col, type in COLUMNS.items() if type == ColumnType.CATEGORICAL
]
NUMERICAL_COLUMNS = [
    col for col, type in COLUMNS.items() if type == ColumnType.NUMERICAL
]
LIST_COLUMNS = [col for col, type in COLUMNS.items() if type == ColumnType.LIST]
