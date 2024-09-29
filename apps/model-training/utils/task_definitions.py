# utils/task_definitions.py

from enum import Enum
from typing import Callable
from utils.database import Match

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
    "towerKills",
    "inhibitorKills",
    "baronKills",
    "dragonKills",
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


def extract_individual_stat(
    match: Match, stat: str, position: str, team_id: str, timestamp: str
):
    try:
        team = match.teams[team_id]
        participant = team["participants"][position]
        return participant["timeline"][timestamp][stat]
    except KeyError:
        return None  # Data missing for this match


def extract_damage(
    match: Match, position: str, team_id: str, damage_type: str, timestamp: str
):
    # 900000, // 15 minutes
    # 1200000, // 20 minutes
    # 1500000, // 25 minutes
    # 1800000, // 30 minutes
    try:
        team = match.teams[team_id]
        participant = team["participants"][position]
        damage_stats = participant["timeline"][timestamp]["damageStats"]
        return damage_stats[damage_type]
    except KeyError:
        return None  # Data missing for this match


def extract_team_stats(match: Match, team_id: str, stat: str, timestamp: str):
    try:
        team = match.teams[team_id]
        return team["teamStats"][timestamp][stat]
    except KeyError:
        return None  # Data missing for this match


# Define tasks
TASKS = {
    "win_prediction": TaskDefinition(
        name="win_prediction",
        task_type=TaskType.BINARY_CLASSIFICATION,
        extractor=extract_win_label,
        weight=0.8,
    ),
    "game_duration": TaskDefinition(
        name="game_duration",
        task_type=TaskType.REGRESSION,
        extractor=extract_game_duration,
        weight=0.01,
    ),
}
for stat in INDIVIDUAL_STATS:
    for position in POSITIONS:
        for team_id in TEAMS:
            for timestamp in TIMESTAMPS:
                task_name = f"{stat}_at_{timestamp}_{position}_{team_id}"
                TASKS[task_name] = TaskDefinition(
                    name=task_name,
                    task_type=TaskType.REGRESSION,
                    extractor=lambda match, pos=position, tid=team_id, ts=timestamp, stat=stat: extract_individual_stat(
                        match, stat, pos, tid, ts
                    ),
                    weight=0.05
                    / (
                        len(INDIVIDUAL_STATS)
                        * len(POSITIONS)
                        * len(TEAMS)
                        * len(TIMESTAMPS)
                    ),
                )

# add damage stats
for damage_type in DAMAGE_STATS:
    for timestamp in TIMESTAMPS:
        for position in POSITIONS:
            for team_id in TEAMS:
                task_name = f"damage_{damage_type}_at_{timestamp}_{position}_{team_id}"
                TASKS[task_name] = TaskDefinition(
                    name=task_name,
                    task_type=TaskType.REGRESSION,
                    extractor=lambda match, pos=position, tid=team_id, dmg_type=damage_type, ts=timestamp: extract_damage(
                        match, pos, tid, dmg_type, ts
                    ),
                    weight=0.05
                    / (
                        len(DAMAGE_STATS)
                        * len(POSITIONS)
                        * len(TEAMS)
                        * len(TIMESTAMPS)
                    ),
                )

# add team stats for all timestamps
for timestamp in TIMESTAMPS:
    for stat in TEAM_STATS:
        for team_id in TEAMS:
            task_name = f"team_stats_{stat}_at_{timestamp}_{team_id}"
            TASKS[task_name] = TaskDefinition(
                name=task_name,
                task_type=TaskType.REGRESSION,
                extractor=lambda match, tid=team_id, stat=stat, ts=timestamp: extract_team_stats(
                    match, tid, stat, ts
                ),
                weight=0.1 / (len(TIMESTAMPS) * len(TEAM_STATS) * len(TEAMS)),
            )
