import { z } from "zod";

export const RegionSchema = z.enum([
  "BR1",
  "EUN1",
  "EUW1",
  "JP1",
  "KR",
  "LA1",
  "LA2",
  "ME1",
  "NA1",
  "OC1",
  "PH2",
  "RU",
  "SG2",
  "TH2",
  "TR1",
  "TW2",
  "VN2",
]);
export type Region = z.infer<typeof RegionSchema>;

export const QueueTypeSchema = z.enum([
  "RANKED_SOLO_5x5",
  "RANKED_FLEX_SR",
  "RANKED_FLEX_TT",
]);
export type QueueType = z.infer<typeof QueueTypeSchema>;

export const TierSchema = z.enum([
  "CHALLENGER",
  "GRANDMASTER",
  "MASTER",
  "DIAMOND",
  "EMERALD",
  "PLATINUM",
  "GOLD",
  "SILVER",
  "BRONZE",
  "IRON",
]);
export type Tier = z.infer<typeof TierSchema>;

export const DivisionSchema = z.enum(["I", "II", "III", "IV"]);
export type Division = z.infer<typeof DivisionSchema>;

export const TierDivisionPairSchema = z.union([
  z.tuple([z.enum(["CHALLENGER", "GRANDMASTER", "MASTER"]), z.literal("I")]),
  z.tuple([
    z.enum([
      "DIAMOND",
      "EMERALD",
      "PLATINUM",
      "GOLD",
      "SILVER",
      "BRONZE",
      "IRON",
    ]),
    DivisionSchema,
  ]),
]);
export type TierDivisionPair = z.infer<typeof TierDivisionPairSchema>;

export const TeamPositionSchema = z.enum(["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]);
export type TeamPosition = z.infer<typeof TeamPositionSchema>;

export const MiniSeriesDTOSchema = z.object({
  losses: z.number(),
  progress: z.string(),
  target: z.number(),
  wins: z.number(),
});

export const LeagueEntryDTOSchema = z.object({
  leagueId: z.string(),
  summonerId: z.string(),
  queueType: QueueTypeSchema,
  tier: TierSchema,
  rank: DivisionSchema,
  leaguePoints: z.number(),
  wins: z.number(),
  losses: z.number(),
  hotStreak: z.boolean(),
  veteran: z.boolean(),
  freshBlood: z.boolean(),
  inactive: z.boolean(),
  miniSeries: MiniSeriesDTOSchema.optional(),
});
export type LeagueEntryDTO = z.infer<typeof LeagueEntryDTOSchema>;

export const SummonerDTOSchema = z.object({
  accountId: z.string().max(56),
  profileIconId: z.number(),
  revisionDate: z.number(),
  id: z.string().max(63),
  puuid: z.string().length(78),
  summonerLevel: z.number(),
});
export type SummonerDTO = z.infer<typeof SummonerDTOSchema>;

export const MatchTypeSchema = z.enum([
  "ranked",
  "normal",
  "tourney",
  "tutorial",
]);
export type MatchType = z.infer<typeof MatchTypeSchema>;

// Match V5
export const MetadataDtoSchema = z.object({
  dataVersion: z.string(),
  matchId: z.string(),
  participants: z.array(z.string()),
});

export const ObjectiveDtoSchema = z.object({
  first: z.boolean(),
  kills: z.number(),
});

export const ObjectivesDtoSchema = z.object({
  baron: ObjectiveDtoSchema,
  champion: ObjectiveDtoSchema,
  dragon: ObjectiveDtoSchema,
  horde: ObjectiveDtoSchema,
  inhibitor: ObjectiveDtoSchema,
  riftHerald: ObjectiveDtoSchema,
  tower: ObjectiveDtoSchema,
});

export const BanDtoSchema = z.object({
  championId: z.number(),
  pickTurn: z.number(),
});

export const ParticipantIdSchema = z.union([
  z.literal(1),
  z.literal(2),
  z.literal(3),
  z.literal(4),
  z.literal(5),
  z.literal(6),
  z.literal(7),
  z.literal(8),
  z.literal(9),
  z.literal(10),
]);
export type ParticipantId = z.infer<typeof ParticipantIdSchema>;

export const TeamIdSchema = z.union([z.literal(100), z.literal(200)]);
export type TeamId = z.infer<typeof TeamIdSchema>;

export const TeamDtoSchema = z.object({
  bans: z.array(BanDtoSchema),
  objectives: ObjectivesDtoSchema,
  teamId: TeamIdSchema,
  win: z.boolean(),
});

export const PerkStatsDtoSchema = z.object({
  defense: z.number(),
  flex: z.number(),
  offense: z.number(),
});

export const PerkStyleSelectionDtoSchema = z.object({
  perk: z.number(),
  var1: z.number(),
  var2: z.number(),
  var3: z.number(),
});

export const PerkStyleDtoSchema = z.object({
  description: z.string(),
  selections: z.array(PerkStyleSelectionDtoSchema),
  style: z.number(),
});

export const PerksDtoSchema = z.object({
  statPerks: PerkStatsDtoSchema,
  styles: z.array(PerkStyleDtoSchema),
});

export const ChallengesDtoSchema = z
  .object({
    "12AssistStreakCount": z.number(),
    baronBuffGoldAdvantageOverThreshold: z.number().optional(),
    controlWardTimeCoverageInRiverOrEnemyHalf: z.number().optional(),
    earliestBaron: z.number().optional(),
    earliestDragonTakedown: z.number().optional(),
    earliestElderDragon: z.number().optional(),
    earlyLaningPhaseGoldExpAdvantage: z.number(),
    fasterSupportQuestCompletion: z.number().optional(),
    fastestLegendary: z.number().optional(),
    hadAfkTeammate: z.number().optional(),
    highestChampionDamage: z.number().optional(),
    highestCrowdControlScore: z.number().optional(),
    highestWardKills: z.number().optional(),
    junglerKillsEarlyJungle: z.number().optional(),
    killsOnLanersEarlyJungleAsJungler: z.number().optional(),
    laningPhaseGoldExpAdvantage: z.number(),
    legendaryCount: z.number(),
    maxCsAdvantageOnLaneOpponent: z.number(),
    maxLevelLeadLaneOpponent: z.number(),
    mostWardsDestroyedOneSweeper: z.number().optional(),
    mythicItemUsed: z.number().optional(),
    playedChampSelectPosition: z.number().optional(),
    soloTurretsLategame: z.number().optional(),
    takedownsFirst25Minutes: z.number().optional(),
    teleportTakedowns: z.number().optional(),
    thirdInhibitorDestroyedTime: z.number().optional(),
    threeWardsOneSweeperCount: z.number().optional(),
    visionScoreAdvantageLaneOpponent: z.number().optional(),
    InfernalScalePickup: z.number(),
    fistBumpParticipation: z.number(),
    voidMonsterKill: z.number(),
    abilityUses: z.number(),
    acesBefore15Minutes: z.number(),
    alliedJungleMonsterKills: z.number(),
    baronTakedowns: z.number(),
    blastConeOppositeOpponentCount: z.number(),
    bountyGold: z.number(),
    buffsStolen: z.number(),
    completeSupportQuestInTime: z.number(),
    controlWardsPlaced: z.number(),
    damagePerMinute: z.number(),
    damageTakenOnTeamPercentage: z.number(),
    dancedWithRiftHerald: z.number(),
    deathsByEnemyChamps: z.number(),
    dodgeSkillShotsSmallWindow: z.number(),
    doubleAces: z.number(),
    dragonTakedowns: z.number(),
    legendaryItemUsed: z.array(z.number()),
    effectiveHealAndShielding: z.number(),
    elderDragonKillsWithOpposingSoul: z.number(),
    elderDragonMultikills: z.number(),
    enemyChampionImmobilizations: z.number(),
    enemyJungleMonsterKills: z.number(),
    epicMonsterKillsNearEnemyJungler: z.number(),
    epicMonsterKillsWithin30SecondsOfSpawn: z.number(),
    epicMonsterSteals: z.number(),
    epicMonsterStolenWithoutSmite: z.number(),
    firstTurretKilled: z.number(),
    firstTurretKilledTime: z.number().optional(),
    flawlessAces: z.number(),
    fullTeamTakedown: z.number(),
    gameLength: z.number(),
    getTakedownsInAllLanesEarlyJungleAsLaner: z.number().optional(),
    goldPerMinute: z.number(),
    hadOpenNexus: z.number(),
    immobilizeAndKillWithAlly: z.number(),
    initialBuffCount: z.number(),
    initialCrabCount: z.number(),
    jungleCsBefore10Minutes: z.number(),
    junglerTakedownsNearDamagedEpicMonster: z.number(),
    kda: z.number(),
    killAfterHiddenWithAlly: z.number(),
    killedChampTookFullTeamDamageSurvived: z.number(),
    killingSprees: z.number(),
    killParticipation: z.number(),
    killsNearEnemyTurret: z.number(),
    killsOnOtherLanesEarlyJungleAsLaner: z.number().optional(),
    killsOnRecentlyHealedByAramPack: z.number(),
    killsUnderOwnTurret: z.number(),
    killsWithHelpFromEpicMonster: z.number(),
    knockEnemyIntoTeamAndKill: z.number(),
    kTurretsDestroyedBeforePlatesFall: z.number(),
    landSkillShotsEarlyGame: z.number(),
    laneMinionsFirst10Minutes: z.number(),
    lostAnInhibitor: z.number(),
    maxKillDeficit: z.number(),
    mejaisFullStackInTime: z.number(),
    moreEnemyJungleThanOpponent: z.number(),
    multiKillOneSpell: z.number(),
    multikills: z.number(),
    multikillsAfterAggressiveFlash: z.number(),
    multiTurretRiftHeraldCount: z.number(),
    outerTurretExecutesBefore10Minutes: z.number(),
    outnumberedKills: z.number(),
    outnumberedNexusKill: z.number(),
    perfectDragonSoulsTaken: z.number(),
    perfectGame: z.number(),
    pickKillWithAlly: z.number(),
    poroExplosions: z.number(),
    quickCleanse: z.number(),
    quickFirstTurret: z.number(),
    quickSoloKills: z.number(),
    riftHeraldTakedowns: z.number(),
    saveAllyFromDeath: z.number(),
    scuttleCrabKills: z.number(),
    shortestTimeToAceFromFirstTakedown: z.number().optional(),
    skillshotsDodged: z.number(),
    skillshotsHit: z.number(),
    snowballsHit: z.number(),
    soloBaronKills: z.number(),
    SWARM_DefeatAatrox: z.number(),
    SWARM_DefeatBriar: z.number(),
    SWARM_DefeatMiniBosses: z.number(),
    SWARM_EvolveWeapon: z.number(),
    SWARM_Have3Passives: z.number(),
    SWARM_KillEnemy: z.number(),
    SWARM_PickupGold: z.number(),
    SWARM_ReachLevel50: z.number(),
    SWARM_Survive15Min: z.number(),
    SWARM_WinWith5EvolvedWeapons: z.number(),
    soloKills: z.number(),
    stealthWardsPlaced: z.number(),
    survivedSingleDigitHpCount: z.number(),
    survivedThreeImmobilizesInFight: z.number(),
    takedownOnFirstTurret: z.number(),
    takedowns: z.number(),
    takedownsAfterGainingLevelAdvantage: z.number(),
    takedownsBeforeJungleMinionSpawn: z.number(),
    takedownsFirstXMinutes: z.number(),
    takedownsInAlcove: z.number(),
    takedownsInEnemyFountain: z.number(),
    teamBaronKills: z.number(),
    teamDamagePercentage: z.number(),
    teamElderDragonKills: z.number(),
    teamRiftHeraldKills: z.number(),
    tookLargeDamageSurvived: z.number(),
    turretPlatesTaken: z.number(),
    turretsTakenWithRiftHerald: z.number(),
    turretTakedowns: z.number(),
    twentyMinionsIn3SecondsCount: z.number(),
    twoWardsOneSweeperCount: z.number(),
    unseenRecalls: z.number(),
    visionScorePerMinute: z.number(),
    wardsGuarded: z.number(),
    wardTakedowns: z.number(),
    wardTakedownsBefore20M: z.number(),
  })
  .partial(); // put into partial because too many are optional

export const MissionsDtoSchema = z.object({
  playerScore0: z.number(),
  playerScore1: z.number(),
  playerScore2: z.number(),
  playerScore3: z.number(),
  playerScore4: z.number(),
  playerScore5: z.number(),
  playerScore6: z.number(),
  playerScore7: z.number(),
  playerScore8: z.number(),
  playerScore9: z.number(),
  playerScore10: z.number(),
  playerScore11: z.number(),
});


export const ParticipantDtoSchema = z.object({
  allInPings: z.number(),
  assistMePings: z.number(),
  assists: z.number(),
  baronKills: z.number(),
  bountyLevel: z.number(),
  champExperience: z.number(),
  champLevel: z.number(),
  championId: z.number(),
  championName: z.string(),
  commandPings: z.number(),
  championTransform: z.number(),
  consumablesPurchased: z.number(),
  challenges: ChallengesDtoSchema,
  damageDealtToBuildings: z.number(),
  damageDealtToObjectives: z.number(),
  damageDealtToTurrets: z.number(),
  damageSelfMitigated: z.number(),
  deaths: z.number(),
  detectorWardsPlaced: z.number(),
  doubleKills: z.number(),
  dragonKills: z.number(),
  eligibleForProgression: z.boolean(),
  enemyMissingPings: z.number(),
  enemyVisionPings: z.number(),
  firstBloodAssist: z.boolean(),
  firstBloodKill: z.boolean(),
  firstTowerAssist: z.boolean(),
  firstTowerKill: z.boolean(),
  gameEndedInEarlySurrender: z.boolean(),
  gameEndedInSurrender: z.boolean(),
  holdPings: z.number(),
  getBackPings: z.number(),
  goldEarned: z.number(),
  goldSpent: z.number(),
  individualPosition: z.string(),
  inhibitorKills: z.number(),
  inhibitorTakedowns: z.number(),
  inhibitorsLost: z.number(),
  item0: z.number(),
  item1: z.number(),
  item2: z.number(),
  item3: z.number(),
  item4: z.number(),
  item5: z.number(),
  item6: z.number(),
  itemsPurchased: z.number(),
  killingSprees: z.number(),
  kills: z.number(),
  lane: z.string(),
  largestCriticalStrike: z.number(),
  largestKillingSpree: z.number(),
  largestMultiKill: z.number(),
  longestTimeSpentLiving: z.number(),
  magicDamageDealt: z.number(),
  magicDamageDealtToChampions: z.number(),
  magicDamageTaken: z.number(),
  missions: MissionsDtoSchema,
  neutralMinionsKilled: z.number(),
  needVisionPings: z.number(),
  nexusKills: z.number(),
  nexusTakedowns: z.number(),
  nexusLost: z.number(),
  objectivesStolen: z.number(),
  objectivesStolenAssists: z.number(),
  onMyWayPings: z.number(),
  participantId: ParticipantIdSchema,
  pentaKills: z.number(),
  perks: PerksDtoSchema,
  physicalDamageDealt: z.number(),
  physicalDamageDealtToChampions: z.number(),
  physicalDamageTaken: z.number(),
  placement: z.number(),
  playerAugment1: z.number(),
  playerAugment2: z.number(),
  playerAugment3: z.number(),
  playerAugment4: z.number(),
  playerSubteamId: z.number(),
  pushPings: z.number(),
  profileIcon: z.number(),
  puuid: z.string(),
  quadraKills: z.number(),
  riotIdGameName: z.string(),
  riotIdTagline: z.string(),
  role: z.string(),
  sightWardsBoughtInGame: z.number(),
  spell1Casts: z.number(),
  spell2Casts: z.number(),
  spell3Casts: z.number(),
  spell4Casts: z.number(),
  subteamPlacement: z.number(),
  summoner1Casts: z.number(),
  summoner1Id: z.number(),
  summoner2Casts: z.number(),
  summoner2Id: z.number(),
  summonerId: z.string(),
  summonerLevel: z.number(),
  summonerName: z.string(),
  teamEarlySurrendered: z.boolean(),
  teamId: TeamIdSchema,
  teamPosition: TeamPositionSchema,
  timeCCingOthers: z.number(),
  timePlayed: z.number(),
  totalAllyJungleMinionsKilled: z.number(),
  totalDamageDealt: z.number(),
  totalDamageDealtToChampions: z.number(),
  totalDamageShieldedOnTeammates: z.number(),
  totalDamageTaken: z.number(),
  totalEnemyJungleMinionsKilled: z.number(),
  totalHeal: z.number(),
  totalHealsOnTeammates: z.number(),
  totalMinionsKilled: z.number(),
  totalTimeCCDealt: z.number(),
  totalTimeSpentDead: z.number(),
  totalUnitsHealed: z.number(),
  tripleKills: z.number(),
  trueDamageDealt: z.number(),
  trueDamageDealtToChampions: z.number(),
  trueDamageTaken: z.number(),
  turretKills: z.number(),
  turretTakedowns: z.number(),
  turretsLost: z.number(),
  unrealKills: z.number(),
  visionScore: z.number(),
  visionClearedPings: z.number(),
  visionWardsBoughtInGame: z.number(),
  wardsKilled: z.number(),
  wardsPlaced: z.number(),
  win: z.boolean(),
});

export const InfoDtoSchema = z.object({
  endOfGameResult: z.string(),
  gameCreation: z.number(),
  gameDuration: z.number(),
  gameEndTimestamp: z.number(),
  gameId: z.number(),
  gameMode: z.string(),
  gameName: z.string(),
  gameStartTimestamp: z.number(),
  gameType: z.string(),
  gameVersion: z.string(),
  mapId: z.number(),
  participants: z.array(ParticipantDtoSchema),
  platformId: z.string(),
  queueId: z.number(),
  teams: z.tuple([TeamDtoSchema, TeamDtoSchema]),
  //teams: z.array(TeamDtoSchema).length(2),
  tournamentCode: z.string().optional(),
});

export const MatchDtoSchema = z.object({
  metadata: MetadataDtoSchema,
  info: InfoDtoSchema,
});

export type MatchDto = z.infer<typeof MatchDtoSchema>;
export type MetadataDto = z.infer<typeof MetadataDtoSchema>;
export type ObjectiveDto = z.infer<typeof ObjectiveDtoSchema>;
export type ObjectivesDto = z.infer<typeof ObjectivesDtoSchema>;
export type BanDto = z.infer<typeof BanDtoSchema>;
export type TeamDto = z.infer<typeof TeamDtoSchema>;
export type PerkStatsDto = z.infer<typeof PerkStatsDtoSchema>;
export type PerkStyleSelectionDto = z.infer<typeof PerkStyleSelectionDtoSchema>;
export type PerkStyleDto = z.infer<typeof PerkStyleDtoSchema>;
export type PerksDto = z.infer<typeof PerksDtoSchema>;
export type ChallengesDto = z.infer<typeof ChallengesDtoSchema>;

export const PositionDtoSchema = z.object({
  x: z.number(),
  y: z.number(),
});

export const DamageStatsDtoSchema = z.object({
  magicDamageDone: z.number(),
  magicDamageDoneToChampions: z.number(),
  magicDamageTaken: z.number(),
  physicalDamageDone: z.number(),
  physicalDamageDoneToChampions: z.number(),
  physicalDamageTaken: z.number(),
  totalDamageDone: z.number(),
  totalDamageDoneToChampions: z.number(),
  totalDamageTaken: z.number(),
  trueDamageDone: z.number(),
  trueDamageDoneToChampions: z.number(),
  trueDamageTaken: z.number(),
});

export const ChampionStatsDtoSchema = z.object({
  abilityHaste: z.number(),
  abilityPower: z.number(),
  armor: z.number(),
  armorPen: z.number(),
  armorPenPercent: z.number(),
  attackDamage: z.number(),
  attackSpeed: z.number(),
  bonusArmorPenPercent: z.number(),
  bonusMagicPenPercent: z.number(),
  ccReduction: z.number(),
  cooldownReduction: z.number(),
  health: z.number(),
  healthMax: z.number(),
  healthRegen: z.number(),
  lifesteal: z.number(),
  magicPen: z.number(),
  magicPenPercent: z.number(),
  magicResist: z.number(),
  movementSpeed: z.number(),
  omnivamp: z.number(),
  physicalVamp: z.number(),
  power: z.number(),
  powerMax: z.number(),
  powerRegen: z.number(),
  spellVamp: z.number(),
});

export const ParticipantFrameDtoSchema = z.object({
  championStats: ChampionStatsDtoSchema,
  currentGold: z.number(),
  damageStats: DamageStatsDtoSchema,
  goldPerSecond: z.number(),
  jungleMinionsKilled: z.number(),
  level: z.number(),
  minionsKilled: z.number(),
  participantId: ParticipantIdSchema,
  position: PositionDtoSchema,
  timeEnemySpentControlled: z.number(),
  totalGold: z.number(),
  xp: z.number(),
});

export const ParticipantFramesDtoSchema = z.record(ParticipantFrameDtoSchema);

const BaseEventSchema = z.object({
  timestamp: z.number(),
  type: z.string(),
});

const ItemEventSchema = BaseEventSchema.extend({
  type: z.enum(["ITEM_PURCHASED", "ITEM_DESTROYED", "ITEM_SOLD", "ITEM_UNDO"]),
  itemId: z.number(),
  participantId: ParticipantIdSchema,
});

const SkillLevelUpEventSchema = BaseEventSchema.extend({
  type: z.literal("SKILL_LEVEL_UP"),
  levelUpType: z.string(),
  participantId: ParticipantIdSchema,
  skillSlot: z.number(),
});

const WardPlacedEventSchema = BaseEventSchema.extend({
  type: z.literal("WARD_PLACED"),
  creatorId: ParticipantIdSchema,
  wardType: z.string(),
});

const LevelUpEventSchema = BaseEventSchema.extend({
  type: z.literal("LEVEL_UP"),
  level: z.number(),
  participantId: ParticipantIdSchema,
});

const PositionSchema = z.object({
  x: z.number(),
  y: z.number(),
});

const BaseChampionSpecialKillEventSchema = BaseEventSchema.extend({
  type: z.literal("CHAMPION_SPECIAL_KILL"),
  killType: z.string(),
  killerId: ParticipantIdSchema,
  position: PositionSchema,
});

const FirstBloodKillEventSchema = BaseChampionSpecialKillEventSchema.extend({
  killType: z.literal("KILL_FIRST_BLOOD"),
});

const MultiKillEventSchema = BaseChampionSpecialKillEventSchema.extend({
  killType: z.literal("KILL_MULTI"),
  multiKillLength: z.number(),
});

const ChampionSpecialKillEventSchema = z.union([
  FirstBloodKillEventSchema,
  MultiKillEventSchema,
  BaseChampionSpecialKillEventSchema,
]);

const DamageDealtSchema = z.object({
  basic: z.boolean(),
  magicDamage: z.number(),
  name: z.string(),
  participantId: ParticipantIdSchema,
  physicalDamage: z.number(),
  spellName: z.string(),
  spellSlot: z.number(),
  trueDamage: z.number(),
  type: z.string(),
});

export const ChampionKillEventSchema = BaseEventSchema.extend({
  type: z.literal("CHAMPION_KILL"),
  bounty: z.number(),
  killStreakLength: z.number(),
  killerId: ParticipantIdSchema,
  assistingParticipantIds: z.array(ParticipantIdSchema).optional(),
  position: PositionSchema,
  shutdownBounty: z.number(),
  victimDamageDealt: z.array(DamageDealtSchema),
  victimDamageReceived: z.array(DamageDealtSchema),
  victimId: ParticipantIdSchema,
});
export type ChampionKillEventDto = z.infer<typeof ChampionKillEventSchema>;

const BaseBuildingKillEventSchema = BaseEventSchema.extend({
  type: z.literal("BUILDING_KILL"),
  bounty: z.number(),
  killerId: ParticipantIdSchema,
  position: PositionSchema,
  teamId: TeamIdSchema,
});

const TowerBuildingKillEventSchema = BaseBuildingKillEventSchema.extend({
  buildingType: z.literal("TOWER_BUILDING"),
  laneType: z.enum(["TOP_LANE", "MID_LANE", "BOT_LANE"]),
  towerType: z.enum([
    "OUTER_TURRET",
    "INNER_TURRET",
    "BASE_TURRET",
    "NEXUS_TURRET",
  ]),
});

const InhibitorBuildingKillEventSchema = BaseBuildingKillEventSchema.extend({
  buildingType: z.literal("INHIBITOR_BUILDING"),
  laneType: z.enum(["TOP_LANE", "MID_LANE", "BOT_LANE"]),
  assistingParticipantIds: z.array(ParticipantIdSchema).optional(),
});

export const BuildingKillEventSchema = z.discriminatedUnion("buildingType", [
  TowerBuildingKillEventSchema,
  InhibitorBuildingKillEventSchema,
]);
export type BuildingKillEventDto = z.infer<typeof BuildingKillEventSchema>;

export const EliteMonsterKillEventSchema = BaseEventSchema.extend({
  type: z.literal("ELITE_MONSTER_KILL"),
  bounty: z.number(),
  killerId: ParticipantIdSchema,
  killerTeamId: TeamIdSchema,
  position: PositionSchema,
  monsterType: z.enum(["DRAGON", "HORDE", "RIFTHERALD", "BARON_NASHOR"]),
  monsterSubType: z.string().optional(),
  assistingParticipantIds: z.array(ParticipantIdSchema).optional(),
});
export type EliteMonsterKillEventDto = z.infer<
  typeof EliteMonsterKillEventSchema
>;

export const EventsTimeLineDtoSchema = z.union([
  ItemEventSchema,
  SkillLevelUpEventSchema,
  WardPlacedEventSchema,
  LevelUpEventSchema,
  ChampionKillEventSchema,
  ChampionSpecialKillEventSchema,
  BuildingKillEventSchema,
  EliteMonsterKillEventSchema,
  BaseEventSchema,
]);

export const FramesTimeLineDtoSchema = z.object({
  events: z.array(EventsTimeLineDtoSchema),
  participantFrames: ParticipantFramesDtoSchema,
  timestamp: z.number(),
});

export const ParticipantTimeLineDtoSchema = z.object({
  participantId: ParticipantIdSchema,
  puuid: z.string(),
});

export const InfoTimeLineDtoSchema = z.object({
  endOfGameResult: z.string(),
  frameInterval: z.number(),
  gameId: z.number(),
  participants: z.array(ParticipantTimeLineDtoSchema),
  frames: z.array(FramesTimeLineDtoSchema),
});

export const MetadataTimeLineDtoSchema = z.object({
  dataVersion: z.string(),
  matchId: z.string(),
  participants: z.array(z.string()),
});

export const TimelineDtoSchema = z.object({
  metadata: MetadataTimeLineDtoSchema,
  info: InfoTimeLineDtoSchema,
});

export type PositionDto = z.infer<typeof PositionDtoSchema>;
export type DamageStatsDto = z.infer<typeof DamageStatsDtoSchema>;
export type ChampionStatsDto = z.infer<typeof ChampionStatsDtoSchema>;
export type ParticipantFrameDto = z.infer<typeof ParticipantFrameDtoSchema>;
export type ParticipantFramesDto = z.infer<typeof ParticipantFramesDtoSchema>;
export type EventsTimeLineDto = z.infer<typeof EventsTimeLineDtoSchema>;
export type FramesTimeLineDto = z.infer<typeof FramesTimeLineDtoSchema>;
export type ParticipantTimeLineDto = z.infer<
  typeof ParticipantTimeLineDtoSchema
>;
export type InfoTimeLineDto = z.infer<typeof InfoTimeLineDtoSchema>;
export type MetadataTimeLineDto = z.infer<typeof MetadataTimeLineDtoSchema>;
export type TimelineDto = z.infer<typeof TimelineDtoSchema>;
