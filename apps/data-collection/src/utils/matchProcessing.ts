import { z } from "zod";
import {
  RiotAPIClient,
  MatchDto,
  TimelineDto,
  ParticipantFrameDto,
  EliteMonsterKillEventDto,
  EventsTimeLineDto,
  ChampionKillEventDto,
  BuildingKillEventDto,
  FramesTimeLineDto,
  ParticipantId,
  TeamId,
  type TeamPosition,
  ParticipantIdSchema,
} from "@draftking/riot-api";

const relevantTimestamps = [
  900000, // 15 minutes
  1200000, // 20 minutes
  1500000, // 25 minutes
  1800000, // 30 minutes
] as const;

const ProcessedParticipantTimelineSchema = z.object({
  level: z.number(),
  kills: z.number(),
  deaths: z.number(),
  assists: z.number(),
  creepScore: z.number(),
  totalGold: z.number(),
  damageStats: z.object({
    magicDamageDoneToChampions: z.number(),
    physicalDamageDoneToChampions: z.number(),
    trueDamageDoneToChampions: z.number(),
  }),
});

const ProcessedTeamStatsSchema = z.object({
  totalKills: z.number(),
  totalDeaths: z.number(),
  totalAssists: z.number(),
  totalGold: z.number(),
  towerKills: z.number(),
  inhibitorKills: z.number(),
  baronKills: z.number(),
  dragonKills: z.number(),
  riftHeraldKills: z.number(),
});
type ProcessedTeamStats = z.infer<typeof ProcessedTeamStatsSchema>;

const ProcessedParticipantSchema = z.object({
  championId: z.number(),
  participantId: ParticipantIdSchema,
  timeline: z.object({
    900000: ProcessedParticipantTimelineSchema,
    1200000: ProcessedParticipantTimelineSchema,
    1500000: ProcessedParticipantTimelineSchema,
    1800000: ProcessedParticipantTimelineSchema,
  }),
});

const ProcessedTeamSchema = z.object({
  win: z.boolean(),
  participants: z.object({
    TOP: ProcessedParticipantSchema,
    JUNGLE: ProcessedParticipantSchema,
    MIDDLE: ProcessedParticipantSchema,
    BOTTOM: ProcessedParticipantSchema,
    UTILITY: ProcessedParticipantSchema,
  }),
  teamStats: z.record(z.string(), ProcessedTeamStatsSchema),
});

const ProcessedMatchDataSchema = z.object({
  gameId: z.number(),
  gameDuration: z.number(),
  gameStartTimestamp: z.number(),
  gameVersionMajorPatch: z.number(),
  gameVersionMinorPatch: z.number(),
  gameVersion: z.string(),
  queueId: z.number(),
  teams: z.object({
    "100": ProcessedTeamSchema,
    "200": ProcessedTeamSchema,
  }),
});

type ProcessedMatchData = z.infer<typeof ProcessedMatchDataSchema>;

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

function initializeProcessedData(matchData: MatchDto): ProcessedMatchData {
  const processedData: ProcessedMatchData = {
    gameId: matchData.info.gameId,
    gameDuration: matchData.info.gameDuration,
    gameStartTimestamp: matchData.info.gameStartTimestamp,
    gameVersion: matchData.info.gameVersion,
    gameVersionMajorPatch: parseInt(
      matchData.info.gameVersion.split(".")[0] ?? "0"
    ),
    gameVersionMinorPatch: parseInt(
      matchData.info.gameVersion.split(".")[1] ?? "0"
    ),
    queueId: matchData.info.queueId,
    teams: {
      100: {
        win: matchData.info.teams[0].win,
        participants: {
          TOP: {
            championId: 0,
            participantId: 0 as ParticipantId,
            timeline: {},
          },
          JUNGLE: {
            championId: 0,
            participantId: 0 as ParticipantId,
            timeline: {},
          },
          MIDDLE: {
            championId: 0,
            participantId: 0 as ParticipantId,
            timeline: {},
          },
          BOTTOM: {
            championId: 0,
            participantId: 0 as ParticipantId,
            timeline: {},
          },
          UTILITY: {
            championId: 0,
            participantId: 0 as ParticipantId,
            timeline: {},
          },
        },
        teamStats: {},
      },
      200: {
        win: matchData.info.teams[1].win,
        participants: {
          TOP: {
            championId: 0,
            participantId: 0 as ParticipantId,
            timeline: {},
          },
          JUNGLE: {
            championId: 0,
            participantId: 0 as ParticipantId,
            timeline: {},
          },
          MIDDLE: {
            championId: 0,
            participantId: 0 as ParticipantId,
            timeline: {},
          },
          BOTTOM: {
            championId: 0,
            participantId: 0 as ParticipantId,
            timeline: {},
          },
          UTILITY: {
            championId: 0,
            participantId: 0 as ParticipantId,
            timeline: {},
          },
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
  const participantIdToTeamPosition: Partial<
    Record<ParticipantId, { teamId: TeamId; teamPosition: TeamPosition }>
  > = {};
  const participantStats: Partial<
    Record<ParticipantId, { kills: number; deaths: number; assists: number }>
  > = {};

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
    const timestamp = relevantTimestamps[i];
    if (timestamp === undefined) {
      throw new Error(`Timestamp ${timestamp} is undefined`);
    }
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
          participantData.timeline[timestamp] =
            participantData.timeline[relevantTimestamps[i - 1]];
        }
      }
    }
  }

  return ProcessedMatchDataSchema.parse(processedData);
}

export { processMatchData, type ProcessedMatchData };
