# utils/match_prediction/task_definitions.py

import pandas as pd
from enum import Enum
from typing import Callable, Dict
from utils.match_prediction.config import TrainingConfig

TEAMS = ["100", "200"]
POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
DAMAGE_STATS = [
    "magicDamageDoneToChampions",
    "physicalDamageDoneToChampions",
    "trueDamageDoneToChampions",
]
TIMESTAMPS = ["900000", "1200000", "1500000", "1800000"]
TEAM_STATS = [
    # "totalKills",
    # "totalDeaths",
    # "totalAssists",
    # "totalGold",
    # "towerKills",
    # "inhibitorKills",
    # "baronKills",
    "dragonKills",
    # "riftHeraldKills",
]

INDIVIDUAL_STATS = [
    "kills",
    # "level",
    "deaths",
    # "assists",
    "totalGold",
    # "creepScore",
]


class TaskType(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


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


# Define tasks
TASKS = {
    "win_prediction": TaskDefinition(
        name="win_prediction",
        getter=get_win_prediction,
        task_type=TaskType.BINARY_CLASSIFICATION,
        weight=0.945,
    ),
    "gameDuration": TaskDefinition(
        name="gameDuration",
        task_type=TaskType.REGRESSION,
        weight=0.01,
    ),
}
for stat in INDIVIDUAL_STATS:
    for position in POSITIONS:
        for team_id in TEAMS:
            for timestamp in TIMESTAMPS:
                task_name = f"team_{team_id}_{position}_{stat}_at_{timestamp}"
                TASKS[task_name] = TaskDefinition(
                    name=task_name,
                    task_type=TaskType.REGRESSION,
                    weight=0.01
                    / (
                        len(INDIVIDUAL_STATS)
                        * len(POSITIONS)
                        * len(TEAMS)
                        * len(TIMESTAMPS)
                    ),
                )

# For damage stats
for damage_type in DAMAGE_STATS:
    for timestamp in TIMESTAMPS:
        for position in POSITIONS:
            for team_id in TEAMS:
                task_name = f"team_{team_id}_{position}_{damage_type}_at_{timestamp}"
                TASKS[task_name] = TaskDefinition(
                    name=task_name,
                    task_type=TaskType.REGRESSION,
                    weight=0.025
                    / (
                        len(DAMAGE_STATS)
                        * len(POSITIONS)
                        * len(TEAMS)
                        * len(TIMESTAMPS)
                    ),
                )

# For team stats
for timestamp in TIMESTAMPS:
    for stat in TEAM_STATS:
        for team_id in TEAMS:
            task_name = f"team_{team_id}_{stat}_at_{timestamp}"
            TASKS[task_name] = TaskDefinition(
                name=task_name,
                task_type=TaskType.REGRESSION,
                weight=0.01 / (len(TIMESTAMPS) * len(TEAM_STATS) * len(TEAMS)),
            )


def get_final_tasks() -> Dict[str, TaskDefinition]:
    """Returns dictionary of final tasks with rebalanced weights (win prediction + total gold)"""
    final_tasks = {
        "win_prediction": TaskDefinition(
            name="win_prediction",
            getter=get_win_prediction,
            task_type=TaskType.BINARY_CLASSIFICATION,
            weight=0.99,  # 99% weight to win prediction
        ),
    }

    # Add total gold tasks for all positions and teams
    gold_tasks_count = len(POSITIONS) * len(TEAMS)  # 5 positions * 2 teams = 10 tasks
    gold_task_weight = 0.01 / gold_tasks_count  # Split 1% among gold tasks

    for position in POSITIONS:
        for team_id in TEAMS:
            for timestamp in TIMESTAMPS:
                task_name = f"team_{team_id}_{position}_totalGold_at_{timestamp}"
                if task_name in TASKS:
                    final_tasks[task_name] = TaskDefinition(
                        name=task_name,
                        task_type=TaskType.REGRESSION,
                        weight=gold_task_weight,
                    )

    return final_tasks


def get_enabled_tasks(config: TrainingConfig, epoch: int) -> Dict[str, TaskDefinition]:
    """Returns dictionary of enabled tasks based on configuration and training phase"""
    if not config.aux_tasks_enabled:
        # Return only win prediction task
        return {
            "win_prediction": TASKS["win_prediction"],
            "gameDuration": TASKS["gameDuration"],
        }
    elif epoch >= config.annealing_epoch:
        # After annealing epoch, use final tasks with rebalanced weights
        return get_final_tasks()
    else:
        # Before annealing epoch, use all tasks
        return TASKS
