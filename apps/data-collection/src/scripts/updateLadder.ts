import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import Bottleneck from "bottleneck";
import { differenceInHours } from "date-fns";
import { sleep } from "../utils";
import {
  TierDivisionPair,
  LeagueEntryDTO,
  RegionSchema,
} from "@draftking/riot-api"; // Adjust the import path as needed
import { RiotAPIClient } from "@draftking/riot-api";
import { PrismaClient } from "@draftking/riot-database";
import { telemetry } from "../utils/telemetry";
import { config } from "dotenv";

config();

const argv = await yargs(hideBin(process.argv))
  .option("region", {
    type: "string",
    demandOption: true,
    describe: "The region to fetch PUUIDs for",
  })
  .parse();

const region = RegionSchema.parse(argv.region);
const apiKey = process.env.X_RIOT_API_KEY;

if (!apiKey) {
  throw new Error("X_RIOT_API_KEY is not set");
}

const riotApiClient = new RiotAPIClient(apiKey, region);
const prisma = new PrismaClient();

const tiersDivisions: TierDivisionPair[] = [
  ["CHALLENGER", "I"],
  ["GRANDMASTER", "I"],
  ["MASTER", "I"],
  ["DIAMOND", "I"],
  ["DIAMOND", "II"],
  ["DIAMOND", "III"],
  ["DIAMOND", "IV"],
  ["EMERALD", "I"],
  ["PLATINUM", "I"],
  ["GOLD", "I"],
  ["SILVER", "I"],
] as const;
const lowEloMaxPages = 120;
const lowEloTiers = ["PLATINUM", "GOLD", "SILVER"];
const queue = "RANKED_SOLO_5x5";

// should be 50 requests every 10 seconds
// we make it 2 times slower to be safe
const limiter = new Bottleneck({
  minTime: 1000,
  reservoir: 50,
  reservoirRefreshAmount: 50,
  reservoirRefreshInterval: 20 * 1000,
  maxConcurrent: 15,
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
          let maxPage = lowEloTiers.includes(tierDivision[0])
            ? lowEloMaxPages
            : 1000;
          while (hasMore && page < maxPage) {
            await limiter.schedule(async () => {
              const leagueEntries = await riotApiClient.getLeagueEntries(
                queue,
                tierDivision,
                page
              );

              if (leagueEntries.length === 0) {
                hasMore = false;
              } else {
                await batchUpsertSummoners(leagueEntries);
                telemetry.trackEvent("LadderUpdated", {
                  count: leagueEntries.length,
                  region,
                  tier: tierDivision[0],
                });
                page++;
              }
            });
          }
        }
        console.log("Ladder updated");
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

async function batchUpsertSummoners(entries: LeagueEntryDTO[]) {
  const batchSize = 100;

  for (let i = 0; i < entries.length; i += batchSize) {
    const batch = entries.slice(i, i + batchSize);

    const batchData = batch.map((entry) => ({
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
    }));

    // Perform batch upsert for the current batch
    await prisma.$transaction(
      batchData.map((data) => prisma.summoner.upsert(data))
    );
  }
}

updateLadder();
