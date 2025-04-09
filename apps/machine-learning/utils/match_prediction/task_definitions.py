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
TEAM_STATS_ALL_TIMESTAMPS = [
    "towerKills",
    "dragonKills",
]

# Stats that don't make sense at 900000 and 1200000
TEAM_STATS_LATE = [
    "baronKills",
    "inhibitorKills",
]

# Stats that don't make sense at 1800000
TEAM_STATS_EARLY_MID = [
    "riftHeraldKills",
]

INDIVIDUAL_STATS = [
    "kills",
    "level",
    "deaths",
    "assists",
    "totalGold",
    "creepScore",
]


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


def get_total_kills_at_20(df: pd.DataFrame) -> pd.Series:
    kills = 0
    timestamp = "1200000"
    for position in POSITIONS:
        for team_id in TEAMS:
            col = f"team_{team_id}_{position}_kills_at_{timestamp}"
            kills += df[col]
    return kills


def get_gold_diff_at_20(df: pd.DataFrame) -> pd.Series:
    timestamp = "1200000"
    stat = "totalGold"
    # blud gold
    blue_gold = 0
    for position in POSITIONS:
        col = f"team_100_{position}_{stat}_at_{timestamp}"
        blue_gold += df[col]
    # red gold
    red_gold = 0
    for position in POSITIONS:
        col = f"team_200_{position}_{stat}_at_{timestamp}"
        red_gold += df[col]
    return blue_gold - red_gold


gold_lead_threshold = 3000  # 3k gold lead


def blue_has_gold_lead_at_20(df: pd.DataFrame) -> pd.Series:
    return (get_gold_diff_at_20(df) >= gold_lead_threshold).astype(int)


def red_has_gold_lead_at_20(df: pd.DataFrame) -> pd.Series:
    return (get_gold_diff_at_20(df) <= -gold_lead_threshold).astype(int)


def gold_is_even_at_20(df: pd.DataFrame) -> pd.Series:
    gold_diff = get_gold_diff_at_20(df)
    return (
        (gold_diff < gold_lead_threshold) & (gold_diff > -gold_lead_threshold)
    ).astype(int)


# Define tasks
TASKS = {
    "win_prediction": TaskDefinition(
        name="win_prediction",
        getter=get_win_prediction,
        task_type=TaskType.BINARY_CLASSIFICATION,
        weight=0.9,
    ),
    "gameDuration": TaskDefinition(
        name="gameDuration",
        task_type=TaskType.REGRESSION,
        weight=0.05,
    ),
    # Add win prediction tasks based on game duration buckets
    "win_prediction_0_20": TaskDefinition(
        name="win_prediction_0_20",
        task_type=TaskType.BINARY_CLASSIFICATION,
        weight=0.1,
    ),
    "win_prediction_20_25": TaskDefinition(
        name="win_prediction_20_25",
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
    "win_prediction_35_40": TaskDefinition(
        name="win_prediction_35_40",
        task_type=TaskType.BINARY_CLASSIFICATION,
        weight=0.1,
    ),
    "win_prediction_40_inf": TaskDefinition(
        name="win_prediction_40_inf",
        task_type=TaskType.BINARY_CLASSIFICATION,
        weight=0.1,
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
                    weight=0.04
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
                    weight=0.04
                    / (
                        len(DAMAGE_STATS)
                        * len(POSITIONS)
                        * len(TEAMS)
                        * len(TIMESTAMPS)
                    ),
                )

# For team stats
for timestamp in TIMESTAMPS:
    # Determine which stats to use based on timestamp
    current_stats = TEAM_STATS_ALL_TIMESTAMPS.copy()
    if timestamp not in ["900000", "1200000"]:
        current_stats.extend(TEAM_STATS_LATE)
    if timestamp != "1800000":
        current_stats.extend(TEAM_STATS_EARLY_MID)

    for stat in current_stats:
        for team_id in TEAMS:
            task_name = f"team_{team_id}_{stat}_at_{timestamp}"
            TASKS[task_name] = TaskDefinition(
                name=task_name,
                task_type=TaskType.REGRESSION,
                weight=0.02
                / (len(TIMESTAMPS) * len(TEAM_STATS_ALL_TIMESTAMPS) * len(TEAMS)),
            )


def get_final_tasks() -> Dict[str, TaskDefinition]:
    """Returns dictionary of final tasks with rebalanced weights (win prediction + total gold)"""
    final_tasks = {
        "win_prediction": TaskDefinition(
            name="win_prediction",
            getter=get_win_prediction,
            task_type=TaskType.BINARY_CLASSIFICATION,
            weight=0.9,
        ),
        # Add win prediction tasks based on game duration buckets
        "win_prediction_0_20": TaskDefinition(
            name="win_prediction_0_20",
            task_type=TaskType.BINARY_CLASSIFICATION,
            weight=0.1,
        ),
        "win_prediction_20_25": TaskDefinition(
            name="win_prediction_20_25",
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
        "win_prediction_35_40": TaskDefinition(
            name="win_prediction_35_40",
            task_type=TaskType.BINARY_CLASSIFICATION,
            weight=0.1,
        ),
        "win_prediction_40_inf": TaskDefinition(
            name="win_prediction_40_inf",
            task_type=TaskType.BINARY_CLASSIFICATION,
            weight=0.1,
        ),
    }

    # Add total gold tasks for all positions and teams
    gold_tasks_count = len(POSITIONS) * len(TEAMS)  # 5 positions * 2 teams = 10 tasks
    gold_task_weight = 0.01 / gold_tasks_count

    for position in POSITIONS:
        for team_id in TEAMS:
            # we only use gold @15 in ui
            for timestamp in ["900000"]:
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
    if epoch >= config.annealing_epoch:
        # After annealing epoch, use final tasks with rebalanced weights
        return get_final_tasks()
    else:
        # Before annealing epoch, use all tasks
        return TASKS
