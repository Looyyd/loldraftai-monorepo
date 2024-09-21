import Bottleneck from "bottleneck";
import { differenceInHours } from "date-fns";
import { sleep } from "../utils";
import { TierDivisionPair, LeagueEntryDTO, Region } from "@draftking/riot-api"; // Adjust the import path as needed
import { RiotAPIClient } from "@draftking/riot-api";
import { PrismaClient } from "@draftking/riot-database";
import { config } from "dotenv";

config();

const region: Region = "EUW1";
const apiKey = process.env.X_RIOT_API_KEY;

if (!apiKey) {
  throw new Error("X_RIOT_API_KEY is not set");
}

const riotApiClient = new RiotAPIClient(apiKey);
const prisma = new PrismaClient();

const tiersDivisions: TierDivisionPair[] = [
  ["CHALLENGER", "I"],
  ["GRANDMASTER", "I"],
  ["MASTER", "I"],
  ["DIAMOND", "I"],
] as const;
const queue = "RANKED_SOLO_5x5";

const limiter = new Bottleneck({
  // Short-term limit: 30 requests every 10 seconds
  minTime: 333, // 10000 ms / 30 = 333.33 ms between requests

  // Long-term limit: 500 requests every 10 minutes
  reservoir: 500, // Start with 500 requests allowed
  reservoirRefreshAmount: 500, // Refill back to 500
  reservoirRefreshInterval: 10 * 60 * 1000, // 10 minutes in milliseconds

  // Optional: Set maximum concurrent requests
  maxConcurrent: 30,
});

let lastUpdate: Date | null = null;

async function updateLadder() {
  try {
    while (true) {
      const now = new Date();
      if (!lastUpdate || differenceInHours(now, lastUpdate) >= 24) {
        console.log("Starting ladder update...");
        lastUpdate = now;
        for (const tierDivision of tiersDivisions) {
          console.log(`Updating ${tierDivision}`);
          let page = 1;
          let hasMore = true;
          while (hasMore) {
            await limiter.schedule(async () => {
              const leagueEntries = await riotApiClient.getLeagueEntries(
                queue,
                tierDivision,
                page
              );

              if (leagueEntries.length === 0) {
                hasMore = false;
              } else {
                for (const entry of leagueEntries) {
                  await upsertSummoner(entry);
                }
                page++;
              }
            });
          }
        }
      } else {
        const hoursUntilNextUpdate = 24 - differenceInHours(now, lastUpdate);
        console.log(`Next update in ${hoursUntilNextUpdate.toFixed(2)} hours.`);
      }

      // Sleep for 1 hour before checking again
      await sleep(1000 * 60 * 60);
    }
  } catch (error) {
    console.error("Error updating ladder:", error);
  } finally {
    await prisma.$disconnect();
  }
}

async function upsertSummoner(entry: LeagueEntryDTO) {
  await prisma.summoner.upsert({
    where: {
      summonerId_region: {
        summonerId: entry.summonerId,
        region: region,
      },
    },
    update: {
      tier: entry.tier,
      rank: entry.rank,
      leaguePoints: entry.leaguePoints,
      rankUpdateTime: new Date(),
    },
    create: {
      summonerId: entry.summonerId,
      region: region,
      tier: entry.tier,
      rank: entry.rank,
      leaguePoints: entry.leaguePoints,
      rankUpdateTime: new Date(),
    },
  });
}

updateLadder();
