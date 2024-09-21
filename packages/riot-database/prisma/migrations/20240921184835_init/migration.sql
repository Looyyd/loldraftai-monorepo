-- CreateEnum
CREATE TYPE "Tier" AS ENUM ('CHALLENGER', 'GRANDMASTER', 'MASTER', 'DIAMOND', 'EMERALD', 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE', 'IRON');

-- CreateEnum
CREATE TYPE "Division" AS ENUM ('I', 'II', 'III', 'IV');

-- CreateEnum
CREATE TYPE "Region" AS ENUM ('BR1', 'EUN1', 'EUW1', 'JP1', 'KR', 'LA1', 'LA2', 'ME1', 'NA1', 'OC1', 'PH2', 'RU', 'SG2', 'TH2', 'TR1', 'TW2', 'VN2');

-- CreateTable
CREATE TABLE "Summoner" (
    "id" TEXT NOT NULL,
    "summonerId" TEXT NOT NULL,
    "region" "Region" NOT NULL,
    "puuid" TEXT,
    "tier" "Tier" NOT NULL,
    "rank" "Division" NOT NULL,
    "leaguePoints" INTEGER NOT NULL,
    "rankUpdateTime" TIMESTAMP(3) NOT NULL,
    "matchesFetchedAt" TIMESTAMP(3),

    CONSTRAINT "Summoner_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Match" (
    "id" TEXT NOT NULL,
    "matchId" TEXT NOT NULL,
    "queueId" INTEGER,
    "region" "Region" NOT NULL,
    "averageTier" "Tier" NOT NULL,
    "averageDivision" "Division" NOT NULL,
    "gameVersionMajorPatch" INTEGER,
    "gameVersionMinorPatch" INTEGER,
    "gameDuration" INTEGER,
    "gameStartTimestamp" TIMESTAMP(3),
    "processed" BOOLEAN NOT NULL DEFAULT false,
    "teams" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Match_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "Summoner_puuid_key" ON "Summoner"("puuid");

-- CreateIndex
CREATE INDEX "Summoner_puuid_idx" ON "Summoner"("puuid");

-- CreateIndex
CREATE INDEX "Summoner_region_idx" ON "Summoner"("region");

-- CreateIndex
CREATE INDEX "Summoner_matchesFetchedAt_idx" ON "Summoner"("matchesFetchedAt");

-- CreateIndex
CREATE UNIQUE INDEX "Summoner_summonerId_region_key" ON "Summoner"("summonerId", "region");

-- CreateIndex
CREATE INDEX "Match_region_idx" ON "Match"("region");

-- CreateIndex
CREATE INDEX "Match_processed_idx" ON "Match"("processed");

-- CreateIndex
CREATE INDEX "Match_gameStartTimestamp_idx" ON "Match"("gameStartTimestamp");

-- CreateIndex
CREATE UNIQUE INDEX "Match_matchId_region_key" ON "Match"("matchId", "region");
