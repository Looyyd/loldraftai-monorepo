# utils/match_prediction/task_definitions.py

import pandas as pd
from enum import Enum
from typing import Callable, Dict
from utils.match_prediction.config import TrainingConfig

TEAMS = ["100", "200"]
POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
TIMESTAMPS = ["900000", "1200000", "1500000", "1800000"]


class TaskType(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"


class TaskDefinition:
    def __init__(
        self,
        name: str,
        task_type: TaskType,
        weight: float,
        getter: Callable[[pd.DataFrame], pd.Series] = None,
    ):
        self.name = name
        self.getter = getter
        self.task_type = task_type
        self.weight = weight


def get_win_prediction(df: pd.DataFrame) -> pd.Series:
    return df["team_100_win"]


def get_game_duration(df: pd.DataFrame) -> pd.Series:
    return df["gameDuration"]


TASKS = {
    "win_prediction": TaskDefinition(
        name="win_prediction",
        getter=get_win_prediction,
        task_type=TaskType.BINARY_CLASSIFICATION,
        weight=1,
    ),
    # Add win prediction tasks based on game duration buckets
    "win_prediction_0_25": TaskDefinition(
        name="win_prediction_0_25",
        task_type=TaskType.BINARY_CLASSIFICATION,
        weight=0.1,
    ),
    "win_prediction_25_30": TaskDefinition(
        name="win_prediction_25_30",
        task_type=TaskType.BINARY_CLASSIFICATION,
        weight=0.1,
    ),
    "win_prediction_30_35": TaskDefinition(
        name="win_prediction_30_35",
        task_type=TaskType.BINARY_CLASSIFICATION,
        weight=0.1,
    ),
    "win_prediction_35_inf": TaskDefinition(
        name="win_prediction_35_inf",
        task_type=TaskType.BINARY_CLASSIFICATION,
        weight=0.1,
    ),
    # TODO : added as a "hack" to make sure the column is added in the dataset, because needed for duration bucketing
    "gameDuration": TaskDefinition(
        name="gameDuration",
        getter=get_game_duration,
        task_type=TaskType.REGRESSION,
        weight=0,
    ),
}

# Add total gold tasks for all positions and teams
gold_tasks_count = len(POSITIONS) * len(TEAMS)  # 5 positions * 2 teams = 10 tasks
gold_task_weight = 0 / gold_tasks_count

for position in POSITIONS:
    for team_id in TEAMS:
        # we only use gold @15 in ui
        for timestamp in ["900000"]:
            task_name = f"team_{team_id}_{position}_totalGold_at_{timestamp}"
            TASKS[task_name] = TaskDefinition(
                name=task_name,
                task_type=TaskType.REGRESSION,
                weight=gold_task_weight,
            )


def get_win_prediction(df: pd.DataFrame) -> pd.Series:
    return df["team_100_win"]


def get_initial_tasks() -> Dict[str, TaskDefinition]:
    # 0 weight for aux tasks, with enough data the model doesn't need them to learn, so we just add them as final tasks because they are used in the frontend
    tasks = TASKS

    for task_name, task_def in tasks.items():
        if task_name == "win_prediction":
            task_def.weight = 1
        else:
            task_def.weight = 0

    return tasks


def get_final_tasks() -> Dict[str, TaskDefinition]:
    return TASKS


def get_enabled_tasks(config: TrainingConfig, epoch: int) -> Dict[str, TaskDefinition]:
    """Returns dictionary of enabled tasks based on configuration and training phase"""
    if epoch >= config.annealing_epoch:
        # After annealing epoch, use final tasks with rebalanced weights
        return get_final_tasks()
    else:
        # Before annealing epoch, use all tasks
        return get_initial_tasks()
