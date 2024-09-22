import Bottleneck from "bottleneck";
import { sleep } from "../utils";
import { Region } from "@draftking/riot-api";
import { RiotAPIClient } from "@draftking/riot-api";
import { PrismaClient } from "@draftking/riot-database";
import { config } from "dotenv";
import { processMatchData } from "../utils/matchProcessing";

config();

const region: Region = "EUW1";
const apiKey = process.env.X_RIOT_API_KEY;

if (!apiKey) {
  throw new Error("X_RIOT_API_KEY is not set");
}

const riotApiClient = new RiotAPIClient(apiKey);
const prisma = new PrismaClient();

// Rate limiter settings based on the API rate limits
const limiter = new Bottleneck({
  minTime: 20, // 20ms between requests (50 requests per second)
  // Limit: 2000 requests every 10 seconds
  reservoir: 2000,
  reservoirRefreshAmount: 2000,
  reservoirRefreshInterval: 10 * 1000, // 10 seconds

  // Adjust maxConcurrent based on your needs and system capabilities
  maxConcurrent: 50,
});

async function processMatches() {
  try {
    while (true) {
      // Fetch unprocessed matches
      const matches = await prisma.match.findMany({
        where: {
          processed: false,
          processingErrored: false,
          region: region,
        },
        take: 1000, // Adjust the batch size as needed
      });

      if (matches.length === 0) {
        console.log("No unprocessed matches found. Sleeping for 1 minute.");
        await sleep(60 * 1000);
        continue;
      }

      console.log(`Found ${matches.length} unprocessed matches.`);

      await Promise.all(
        matches.map((match) =>
          limiter.schedule(async () => {
            try {
              // TODO: mark as processed if failed? or have failed flag?
              const processedData = await processMatchData(
                riotApiClient,
                match.matchId
              );

              // Update the match record with processed data
              await prisma.match.update({
                where: {
                  id: match.id,
                },
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

              console.log(`Processed match ${match.matchId}`);
            } catch (error) {
              console.error(`Error processing match ${match.matchId}:`, error);
              // Mark the match as processed but with an error
              await prisma.match.update({
                where: {
                  id: match.id,
                },
                data: {
                  processed: true,
                  processingErrored: true,
                },
              });
            }
          })
        )
      );
    }
  } catch (error) {
    console.error("Error in processMatches:", error);
  } finally {
    await prisma.$disconnect();
  }
}

processMatches();
