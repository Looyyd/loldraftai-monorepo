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
  ParticipantId,
  TeamId,
  type TeamPosition,
} from "@draftking/riot-api";

interface ParticipantTimelineData {
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
      participants: Record<
        TeamPosition,
        {
          championId: number;
          participantId: ParticipantId;
          timeline: Record<number, ParticipantTimelineData>;
        }
      >;
      teamStats: Record<number, ProcessedTeamStats>;
    }
  >;
}

function initializeProcessedData(matchData: MatchDto): ProcessedMatchData {
  const processedData: ProcessedMatchData = {
    gameId: matchData.info.gameId,
    gameDuration: matchData.info.gameDuration,
    gameVersion: matchData.info.gameVersion,
    queueId: matchData.info.queueId,
    teams: {
      100: {
        win: matchData.info.teams[0].win,
        objectives: matchData.info.teams[0].objectives,
        participants: {
          TOP: { championId: 0, participantId: 0 as ParticipantId, timeline: {} },
          JUNGLE: { championId: 0, participantId: 0 as ParticipantId, timeline: {} },
          MIDDLE: { championId: 0, participantId: 0 as ParticipantId, timeline: {} },
          BOTTOM: { championId: 0, participantId: 0 as ParticipantId, timeline: {} },
          UTILITY: { championId: 0, participantId: 0 as ParticipantId, timeline: {} },
        },
        teamStats: {},
      },
      200: {
        win: matchData.info.teams[1].win,
        objectives: matchData.info.teams[1].objectives,
        participants: {
          TOP: { championId: 0, participantId: 0 as ParticipantId, timeline: {} },
          JUNGLE: { championId: 0, participantId: 0 as ParticipantId, timeline: {} },
          MIDDLE: { championId: 0, participantId: 0 as ParticipantId, timeline: {} },
          BOTTOM: { championId: 0, participantId: 0 as ParticipantId, timeline: {} },
          UTILITY: { championId: 0, participantId: 0 as ParticipantId, timeline: {} },
        },
        teamStats: {},
      },
    },
  };

  return processedData;
}

function initializeParticipantsData(
  matchData: MatchDto,
  processedData: ProcessedMatchData
): {
  participantIdToTeamPosition: Record<
    ParticipantId,
    { teamId: TeamId; teamPosition: TeamPosition }
  >;
  participantStats: Record<
    ParticipantId,
    { kills: number; deaths: number; assists: number }
  >;
} {
  const participantIdToTeamPosition: Partial<Record<
    ParticipantId,
    { teamId: TeamId; teamPosition: TeamPosition }
  >> = {};
  const participantStats: Partial<Record<
    ParticipantId,
    { kills: number; deaths: number; assists: number }
  >> = {};

  matchData.info.participants.forEach((participant) => {
    const teamId = participant.teamId as TeamId;
    const teamPosition = participant.teamPosition as TeamPosition;
    const participantId = participant.participantId as ParticipantId;

    participantIdToTeamPosition[participantId] = {
      teamId,
      teamPosition,
    };

    processedData.teams[teamId].participants[teamPosition] = {
      championId: participant.championId,
      participantId: participantId,
      timeline: {},
    };

    participantStats[participantId] = {
      kills: 0,
      deaths: 0,
      assists: 0,
    };
  });

  return {
    participantIdToTeamPosition: participantIdToTeamPosition as Record<
      ParticipantId,
      { teamId: TeamId; teamPosition: TeamPosition }
    >,
    participantStats: participantStats as Record<
      ParticipantId,
      { kills: number; deaths: number; assists: number }
    >,
  };
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

function processEvents(
  events: EventsTimeLineDto[],
  participantStats: Record<
    ParticipantId,
    { kills: number; deaths: number; assists: number }
  >,
  participantIdToTeamPosition: Record<
    ParticipantId,
    { teamId: TeamId; teamPosition: TeamPosition }
  >,
  teamStats: Record<TeamId, ProcessedTeamStats>
) {
  events.forEach((event) => {
    if (event.type === "CHAMPION_KILL") {
      const e = event as ChampionKillEventDto;
      const killerId = e.killerId as ParticipantId;
      const victimId = e.victimId as ParticipantId;

      if (killerId && participantStats[killerId]) {
        participantStats[killerId].kills++;
        const teamId = participantIdToTeamPosition[killerId].teamId;
        teamStats[teamId].totalKills++;
      }
      if (victimId && participantStats[victimId]) {
        participantStats[victimId].deaths++;
        const teamId = participantIdToTeamPosition[victimId].teamId;
        teamStats[teamId].totalDeaths++;
      }
      e.assistingParticipantIds?.forEach((assistId) => {
        if (participantStats[assistId]) {
          participantStats[assistId].assists++;
          const teamId = participantIdToTeamPosition[assistId].teamId;
          teamStats[teamId].totalAssists++;
        }
      });
    } else if (event.type === "BUILDING_KILL") {
      const e = event as BuildingKillEventDto;
      const teamId = e.teamId as TeamId;
      if (e.buildingType === "TOWER_BUILDING") {
        teamStats[teamId].towerKills++;
      } else if (e.buildingType === "INHIBITOR_BUILDING") {
        teamStats[teamId].inhibitorKills++;
      }
    } else if (event.type === "ELITE_MONSTER_KILL") {
      const e = event as EliteMonsterKillEventDto;
      const killerTeamId = e.killerTeamId as TeamId;
      switch (e.monsterType) {
        case "BARON_NASHOR":
          teamStats[killerTeamId].baronKills++;
          break;
        case "DRAGON":
          teamStats[killerTeamId].dragonKills++;
          break;
        case "RIFTHERALD":
          teamStats[killerTeamId].riftHeraldKills++;
          break;
      }
    }
  });
}

function processTimelineFrame(
  frame: FramesTimeLineDto,
  timestamp: number,
  processedData: ProcessedMatchData,
  participantIdToTeamPosition: Record<
    ParticipantId,
    { teamId: TeamId; teamPosition: TeamPosition }
  >,
  participantStats: Record<
    ParticipantId,
    { kills: number; deaths: number; assists: number }
  >,
  teamStats: Record<TeamId, ProcessedTeamStats>
) {
  // Reset team totalGold for this frame
  teamStats[100].totalGold = 0;
  teamStats[200].totalGold = 0;

  Object.entries(frame.participantFrames).forEach(
    ([participantIdStr, participantFrame]: [string, ParticipantFrameDto]) => {
      const participantId = parseInt(participantIdStr) as ParticipantId;
      const { teamId, teamPosition } =
        participantIdToTeamPosition[participantId];

      const participantData =
        processedData.teams[teamId].participants[teamPosition];

      participantData.timeline[timestamp] = {
        level: participantFrame.level,
        kills: participantStats[participantId].kills,
        deaths: participantStats[participantId].deaths,
        assists: participantStats[participantId].assists,
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

      // Accumulate team totalGold
      teamStats[teamId].totalGold += participantFrame.totalGold;
    }
  );

  // Deep copy teamStats into processedData
  processedData.teams[100].teamStats[timestamp] = { ...teamStats[100] };
  processedData.teams[200].teamStats[timestamp] = { ...teamStats[200] };
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

  // Round timestamp to the nearest frame interval (which is 60000ms)
  const frameInterval = timelineData.info.frameInterval;
  timelineData.info.frames.forEach((frame) => {
    frame.timestamp =
      Math.round(frame.timestamp / frameInterval) * frameInterval;
  });

  const processedData = initializeProcessedData(matchData);

  const { participantIdToTeamPosition, participantStats } =
    initializeParticipantsData(matchData, processedData);

  const teamStats = initializeTeamStats();

  const relevantTimestamps = [
    minutesToMs(15),
    minutesToMs(20),
    minutesToMs(25),
    minutesToMs(30),
  ] as const;

  timelineData.info.frames.forEach((frame) => {
    processEvents(
      frame.events,
      participantStats,
      participantIdToTeamPosition,
      teamStats
    );

    if (relevantTimestamps.includes(frame.timestamp)) {
      processTimelineFrame(
        frame,
        frame.timestamp,
        processedData,
        participantIdToTeamPosition,
        participantStats,
        teamStats
      );
    }
  });

  // If first relevant timestamp is not included, throw an error
  if (!(relevantTimestamps[0] in processedData.teams[100].teamStats)) {
    throw new Error(
      `First relevant timestamp ${relevantTimestamps[0]} is not included in processed data`
    );
  }

  // If not all relevant timestamps are included, duplicate the last valid timestamp
  for (let i = 1; i < relevantTimestamps.length; i++) {
    const timestamp = relevantTimestamps[i] as number;
    if (!(timestamp in processedData.teams[100].teamStats)) {
      processedData.teams[100].teamStats[timestamp] = processedData.teams[100]
        .teamStats[relevantTimestamps[i - 1] as number] as ProcessedTeamStats;
      processedData.teams[200].teamStats[timestamp] = processedData.teams[200]
        .teamStats[relevantTimestamps[i - 1] as number] as ProcessedTeamStats;

      for (const teamId of [100, 200] as const) {
        for (const teamPosition in processedData.teams[teamId].participants) {
          const participantData =
            processedData.teams[teamId].participants[
              teamPosition as TeamPosition
            ];
          participantData.timeline[timestamp] = participantData.timeline[
            relevantTimestamps[i - 1] as number
          ] as ParticipantTimelineData;
        }
      }
    }
  }

  return processedData;
}

export { processMatchData, type ProcessedMatchData };
