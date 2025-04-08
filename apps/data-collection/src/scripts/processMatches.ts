import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import Bottleneck from "bottleneck";
import { sleep } from "../utils";
import { RegionSchema } from "@draftking/riot-api";
import { RiotAPIClient } from "@draftking/riot-api";
import { Match, PrismaClient } from "@draftking/riot-database";
import { config } from "dotenv";
import { processMatchData } from "../utils/matchProcessing";
import { telemetry } from "../utils/telemetry";

config();

// Enhance logging to always show timestamp and add log levels
const log = (level: "INFO" | "ERROR" | "DEBUG", message: string) => {
  console.log(`[${new Date().toISOString()}] [${level}] ${message}`);
};

const argv = await yargs(hideBin(process.argv))
  .option("region", {
    type: "string",
    demandOption: true,
    describe: "The region to fetch PUUIDs for",
  })
  .parse();

log("INFO", `Starting processMatches for region: ${argv.region}`);

const region = RegionSchema.parse(argv.region);
const apiKey = process.env.X_RIOT_API_KEY;

if (!apiKey) {
  throw new Error("X_RIOT_API_KEY is not set");
}

const riotApiClient = new RiotAPIClient(apiKey, region);
const prisma = new PrismaClient();

// Rate limiter settings based on the API rate limits
const limiter = new Bottleneck({
  minTime: 100,
  // Limit: 2000 requests every 10 seconds, but we'll use 500 to be safe
  reservoir: 250,
  reservoirRefreshAmount: 250,
  reservoirRefreshInterval: 10 * 1000, // 10 seconds

  // we limit the number of concurrent, to avoid having too many that trigger when a rate limit await is activated
  // this happens when server says to wait before retrying
  maxConcurrent: 10,
});

let isShuttingDown = false;

// Add signal handlers
process.on("SIGTERM", () => {
  log("INFO", "Received SIGTERM signal");
  isShuttingDown = true;
});

process.on("SIGINT", () => {
  log("INFO", "Received SIGINT signal");
  isShuttingDown = true;
});

async function processMatches() {
  log("INFO", "processMatches function started");
  try {
    let nextMatchesPromise: Promise<Match[]> | null = null;
    let iterationCount = 0;

    while (!isShuttingDown) {
      iterationCount++;
      log(
        "DEBUG",
        `Starting iteration ${iterationCount} of main processing loop`
      );

      try {
        // Log the start of match fetching
        log("DEBUG", "Fetching next batch of matches");

        const matches = await (nextMatchesPromise ??
          (prisma.$queryRaw`
          SELECT *
          FROM "Match"
          WHERE processed = false 
          AND "processingErrored" = false
          AND region = ${region}::text::"Region"
          LIMIT 500
        ` as Promise<Match[]>));

        log("INFO", `Found ${matches.length} matches to process`);

        // Start fetching next batch immediately
        const currentlyProcessingIds = new Set(matches.map((m) => m.id));
        log(
          "DEBUG",
          `Pre-fetching next batch, excluding ${currentlyProcessingIds.size} currently processing IDs`
        );

        nextMatchesPromise = prisma.$queryRaw`
          SELECT *
          FROM "Match"
          WHERE processed = false 
          AND "processingErrored" = false
          AND region = ${region}::text::"Region"
          AND id NOT IN (${Array.from(currentlyProcessingIds)})
          LIMIT 500
        ` as Promise<Match[]>;

        if (matches.length === 0) {
          log("INFO", "No matches found, sleeping for 60 seconds");
          await sleep(60 * 1000);
          log("DEBUG", "Waking up from sleep");
          continue;
        }

        // Process matches in parallel
        log(
          "DEBUG",
          `Starting parallel processing of ${matches.length} matches`
        );
        const startTime = Date.now();

        await Promise.all(
          matches.map(async (match) => {
            log("DEBUG", `Processing match ${match.matchId}`);

            try {
              const apiStart = Date.now();
              const processedData = await processMatchData(
                riotApiClient,
                match.matchId,
                limiter
              );
              log(
                "DEBUG",
                `API call completed in ${Date.now() - apiStart}ms for match ${
                  match.matchId
                }`
              );

              if (
                processedData.queueId !== 420 &&
                processedData.queueId !== 700
              ) {
                log(
                  "INFO",
                  `Skipping non-ranked/clash match ${match.matchId} (queueId: ${processedData.queueId})`
                );
                await prisma.match.update({
                  where: { id: match.id },
                  data: {
                    processed: true,
                    processingErrored: true,
                    queueId: processedData.queueId,
                  },
                });
                return;
              }

              const updateStart = Date.now();
              await prisma.match.update({
                where: { id: match.id },
                data: {
                  processed: true,
                  gameDuration: processedData.gameDuration,
                  gameStartTimestamp: new Date(
                    processedData.gameStartTimestamp
                  ),
                  queueId: processedData.queueId,
                  gameVersionMajorPatch: processedData.gameVersionMajorPatch,
                  gameVersionMinorPatch: processedData.gameVersionMinorPatch,
                  teams: processedData.teams,
                },
              });
              log(
                "DEBUG",
                `Database update completed in ${
                  Date.now() - updateStart
                }ms for match ${match.matchId}`
              );

              telemetry.trackEvent("MatchesProcessed", {
                count: 1,
                region,
              });
            } catch (error) {
              log("ERROR", `Error processing match ${match.matchId}: ${error}`);

              try {
                await prisma.match.update({
                  where: { id: match.id },
                  data: {
                    processed: true,
                    processingErrored: true,
                  },
                });
                log("DEBUG", `Marked match ${match.matchId} as errored`);
              } catch (updateError) {
                log(
                  "ERROR",
                  `Failed to mark match ${match.matchId} as errored: ${updateError}`
                );
              }
            }
          })
        );

        log(
          "INFO",
          `Completed batch processing in ${Date.now() - startTime}ms`
        );
      } catch (batchError) {
        log("ERROR", `Error processing batch: ${batchError}`);
        // Sleep briefly before retrying to prevent rapid failure loops
        await sleep(5000);
      }
    }
  } catch (error) {
    log("ERROR", `Fatal error in processMatches: ${error}`);
  } finally {
    log("INFO", "Disconnecting from database");
    await prisma.$disconnect();
    log("INFO", "Database disconnected");
  }
}

// Properly handle the promise
log("INFO", "Starting main process");
processMatches().catch((error) => {
  log("ERROR", `Unhandled error in main process: ${error}`);
  process.exitCode = 1;
});
