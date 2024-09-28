# utils/task_definitions.py

from enum import Enum
from typing import Callable
from utils.database import Match

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


class TaskType(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


class TaskDefinition:
    def __init__(
        self, name: str, task_type: TaskType, extractor: Callable, weight: float
    ):
        self.name = name
        self.task_type = task_type
        self.extractor = extractor
        self.weight = weight


def extract_win_label(match: Match):
    return 1 if match.teams.get("100", {}).get("win", False) else 0


def extract_game_duration(match: Match):
    return match.gameDuration


def extract_gold_at_15(match: Match, position: str, team_id: str):
    try:
        team = match.teams[team_id]
        participant = team["participants"][position]
        total_gold = participant["timeline"]["900000"]["totalGold"]
        return total_gold
    except KeyError:
        return None  # Data missing for this match


# Define tasks
TASKS = {
    "win_prediction": TaskDefinition(
        name="win_prediction",
        task_type=TaskType.BINARY_CLASSIFICATION,
        extractor=extract_win_label,
        weight=0.7,
    ),
    "game_duration": TaskDefinition(
        name="game_duration",
        task_type=TaskType.REGRESSION,
        extractor=extract_game_duration,
        weight=0.2,
    ),
}
# Add gold_at_15 tasks for each position and team
for position in POSITIONS:
    for team_id in ["100", "200"]:
        task_name = f"gold_at_15_{position}_{team_id}"
        TASKS[task_name] = TaskDefinition(
            name=task_name,
            task_type=TaskType.REGRESSION,
            extractor=lambda match, pos=position, tid=team_id: extract_gold_at_15(
                match, pos, tid
            ),
            weight=0.01,  # You may want to adjust this weight
        )
