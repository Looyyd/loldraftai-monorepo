import {
  RiotAPIClient,
  MatchDto,
  TimelineDto,
  ParticipantFrameDto,
  ObjectivesDto,
  EliteMonsterKillEventDto,
  EventsTimeLineDto,
  ChampionKillEventDto,
  BuildingKillEventDto,
  FramesTimeLineDto,
  ParticipantIdSchema,
  ParticipantId,
  TeamId,
  type TeamPosition,
} from "@draftking/riot-api";
interface ProcessedParticipantData {
  championId: number;
  teamPosition: TeamPosition;
  level: number;
  kills: number;
  deaths: number;
  assists: number;
  creepScore: number;
  totalGold: number;
  damageStats: {
    magicDamageDoneToChampions: number;
    physicalDamageDoneToChampions: number;
    trueDamageDoneToChampions: number;
  };
}

interface ProcessedTeamStats {
  totalKills: number;
  totalDeaths: number;
  totalAssists: number;
  totalGold: number;
  towerKills: number;
  inhibitorKills: number;
  baronKills: number;
  dragonKills: number;
  riftHeraldKills: number;
}

interface ProcessedTimelineData {
  participants: Record<ParticipantId, ProcessedParticipantData>;
  teamStats: Record<TeamId, ProcessedTeamStats>;
}

interface ProcessedMatchData {
  gameId: number;
  gameDuration: number;
  gameVersion: string;
  queueId: number;
  teams: Record<
    TeamId,
    {
      win: boolean;
      objectives: ObjectivesDto;
    }
  >;
  timeline: Record<number, ProcessedTimelineData>;
}

type ParticipantStats = Record<
  ParticipantId,
  {
    championId: number;
    teamId: TeamId;
    kills: number;
    deaths: number;
    assists: number;
    teamPosition: TeamPosition;
  }
>;

function initializeProcessedData(matchData: MatchDto): ProcessedMatchData {
  const processedData = {
    gameId: matchData.info.gameId,
    gameDuration: matchData.info.gameDuration,
    gameVersion: matchData.info.gameVersion,
    queueId: matchData.info.queueId,
    teams: {
      100: {
        win: matchData.info.teams[0].win,
        objectives: matchData.info.teams[0].objectives,
      },
      200: {
        win: matchData.info.teams[1].win,
        objectives: matchData.info.teams[1].objectives,
      },
    },
    timeline: {},
  };

  matchData.info.teams.forEach((team) => {
    processedData.teams[team.teamId] = {
      win: team.win,
      objectives: team.objectives,
    };
  });

  return processedData;
}

function initializeParticipantStats(matchData: MatchDto): ParticipantStats {
  const participantStats = {} as ParticipantStats;
  matchData.info.participants.forEach((participant) => {
    participantStats[participant.participantId] = {
      championId: participant.championId,
      teamId: participant.teamId,
      teamPosition: participant.teamPosition,
      kills: 0,
      deaths: 0,
      assists: 0,
    };
  });
  return participantStats;
}

function initializeTeamStats(): Record<TeamId, ProcessedTeamStats> {
  return {
    100: {
      totalKills: 0,
      totalDeaths: 0,
      totalAssists: 0,
      totalGold: 0,
      towerKills: 0,
      inhibitorKills: 0,
      baronKills: 0,
      dragonKills: 0,
      riftHeraldKills: 0,
    },
    200: {
      totalKills: 0,
      totalDeaths: 0,
      totalAssists: 0,
      totalGold: 0,
      towerKills: 0,
      inhibitorKills: 0,
      baronKills: 0,
      dragonKills: 0,
      riftHeraldKills: 0,
    },
  };
}

function updateChampionKillStats(
  event: ChampionKillEventDto,
  participantStats: ParticipantStats,
  teamStats: Record<TeamId, ProcessedTeamStats>
) {
  if (participantStats[event.killerId]) {
    participantStats[event.killerId].kills++;
    teamStats[participantStats[event.killerId].teamId].totalKills++;
  }
  if (participantStats[event.victimId]) {
    participantStats[event.victimId].deaths++;
    teamStats[participantStats[event.victimId].teamId].totalDeaths++;
  }
  event.assistingParticipantIds?.forEach((assistId) => {
    if (participantStats[assistId]) {
      participantStats[assistId].assists++;
      teamStats[participantStats[assistId].teamId].totalAssists++;
    }
  });
}

function updateBuildingKillStats(
  event: BuildingKillEventDto,
  teamStats: Record<TeamId, ProcessedTeamStats>
) {
  if (event.buildingType === "TOWER_BUILDING") {
    teamStats[event.teamId].towerKills++;
  } else if (event.buildingType === "INHIBITOR_BUILDING") {
    teamStats[event.teamId].inhibitorKills++;
  }
}

function updateEliteMonsterKillStats(
  event: EliteMonsterKillEventDto,
  teamStats: Record<TeamId, ProcessedTeamStats>
) {
  switch (event.monsterType) {
    case "BARON_NASHOR":
      teamStats[event.killerTeamId].baronKills++;
      break;
    case "DRAGON":
      teamStats[event.killerTeamId].dragonKills++;
      break;
    case "RIFTHERALD":
      teamStats[event.killerTeamId].riftHeraldKills++;
      break;
  }
}

function processEvents(
  events: EventsTimeLineDto[],
  participantStats: ParticipantStats,
  teamStats: Record<TeamId, ProcessedTeamStats>
) {
  events.forEach((event) => {
    if (event.type === "CHAMPION_KILL") {
      updateChampionKillStats(
        event as ChampionKillEventDto,
        participantStats,
        teamStats
      );
    } else if (event.type === "BUILDING_KILL") {
      updateBuildingKillStats(event as BuildingKillEventDto, teamStats);
    } else if (event.type === "ELITE_MONSTER_KILL") {
      updateEliteMonsterKillStats(event as EliteMonsterKillEventDto, teamStats);
    }
  });
}

function processTimelineFrame(
  frame: FramesTimeLineDto,
  timestamp: number,
  processedData: ProcessedMatchData,
  participantStats: ParticipantStats,
  teamStats: Record<TeamId, ProcessedTeamStats>
) {
  processedData.timeline[timestamp] = {
    participants: {} as Record<ParticipantId, ProcessedParticipantData>,
    teamStats: JSON.parse(JSON.stringify(teamStats)), // Deep copy current team stats
  };

  Object.entries(frame.participantFrames).forEach(
    ([participantId, participantFrame]: [string, ParticipantFrameDto]) => {
      const pId = ParticipantIdSchema.parse(parseInt(participantId));
      // We know timestamp exists because we set it just before
      processedData.timeline[timestamp]!.participants[pId] = {
        championId: participantStats[pId].championId,
        teamPosition: participantStats[pId].teamPosition,
        level: participantFrame.level,
        kills: participantStats[pId].kills,
        deaths: participantStats[pId].deaths,
        assists: participantStats[pId].assists,
        creepScore:
          participantFrame.minionsKilled + participantFrame.jungleMinionsKilled,
        totalGold: participantFrame.totalGold,
        damageStats: {
          magicDamageDoneToChampions:
            participantFrame.damageStats.magicDamageDoneToChampions,
          physicalDamageDoneToChampions:
            participantFrame.damageStats.physicalDamageDoneToChampions,
          trueDamageDoneToChampions:
            participantFrame.damageStats.trueDamageDoneToChampions,
        },
      };

      // Update team gold
      const teamId = participantStats[pId].teamId;
      // We know timestamp exists because we set it just before
      processedData.timeline[timestamp]!.teamStats[teamId].totalGold +=
        participantFrame.totalGold;
    }
  );
}

function minutesToMs(minutes: number): number {
  return minutes * 60 * 1000;
}

async function processMatchData(
  client: RiotAPIClient,
  matchId: string
): Promise<ProcessedMatchData> {
  const matchData: MatchDto = await client.getMatchById(matchId);
  const timelineData: TimelineDto = await client.getMatchTimelineById(matchId);

  // round timestamp to the nearest frame interval(which is 60000ms)
  const frameInterval = timelineData.info.frameInterval;
  timelineData.info.frames.forEach((frame) => {
    frame.timestamp =
      Math.round(frame.timestamp / frameInterval) * frameInterval;
  });

  const processedData = initializeProcessedData(matchData);
  const participantStats = initializeParticipantStats(matchData);
  const teamStats = initializeTeamStats();

  const relevantTimestamps = [
    minutesToMs(15),
    minutesToMs(20),
    minutesToMs(25),
    minutesToMs(30),
  ] as const;

  timelineData.info.frames.forEach((frame) => {
    if (relevantTimestamps.includes(frame.timestamp)) {
      processTimelineFrame(
        frame,
        frame.timestamp,
        processedData,
        participantStats,
        teamStats
      );
    }

    processEvents(frame.events, participantStats, teamStats);
  });

  // if first relevant timestamp is not included, throw an error
  if (!(relevantTimestamps[0] in processedData.timeline)) {
    throw new Error(
      `First relevant timestamp ${relevantTimestamps[0]} is not included in processed data`
    );
  }
  // TODO: should get from last available timestamp instead
  // If note all relevant timestamps are included, duplicate the last valid timestamp
  for (let i = 1; i < relevantTimestamps.length; i++) {
    const timestamp = relevantTimestamps[i] as number; // safe because iterate over length
    if (!(timestamp in processedData.timeline)) {
      processedData.timeline[timestamp] = processedData.timeline[
        relevantTimestamps[i - 1] as number // safe because iterate over length
      ] as ProcessedTimelineData; // exists because 0 exists and we set all future timestamps
    }
  }

  return processedData;
}

// TODO: order by teamPosition

export { processMatchData, type ProcessedMatchData };
