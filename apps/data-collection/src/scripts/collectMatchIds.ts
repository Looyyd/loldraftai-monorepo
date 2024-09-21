import Bottleneck from "bottleneck";
import { sleep } from "../utils";
import { Region } from "@draftking/riot-api";
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

// Rate limiter settings based on the API rate limits
const limiter = new Bottleneck({
  // Limit: 2000 requests every 10 seconds
  reservoir: 500,
  reservoirRefreshAmount: 2000,
  reservoirRefreshInterval: 10 * 1000, // 10 seconds

  // Adjust maxConcurrent based on your needs and system capabilities
  maxConcurrent: 50,
});

async function collectMatchIds() {
  try {
    while (true) {
      // Fetch summoners with a PUUID and outdated matchesFetchedAt
      const summoners = await prisma.summoner.findMany({
        where: {
          puuid: {
            not: null,
          },
          region: region,
          OR: [
            { matchesFetchedAt: null },
            {
              matchesFetchedAt: {
                lt: new Date(Date.now() - 24 * 60 * 60 * 1000), // Older than 24 hours
              },
            },
          ],
        },
        take: 1000, // Adjust the batch size as needed
      });

      if (summoners.length === 0) {
        console.log(
          "No summoners to fetch match IDs for. Sleeping for 1 minute."
        );
        await sleep(60 * 1000);
        continue;
      }

      console.log(`Found ${summoners.length} summoners to fetch match IDs.`);

      await Promise.all(
        summoners.map((summoner) =>
          limiter.schedule(async () => {
            try {
              const matchIds = await riotApiClient.getMatchIdsByPuuid(
                summoner.puuid!,
                {
                  type: "ranked",
                  queue: 420, // Ranked Solo/Duo queue
                  count: 100, // max count
                }
              );

              for (const matchId of matchIds) {
                await prisma.match.upsert({
                  where: {
                    matchId_region: {
                      matchId: matchId,
                      region: region,
                    },
                  },
                  update: {},
                  create: {
                    matchId: matchId,
                    region: region,
                    processed: false,
                    // We don't do average, instead we use the summoner's tier and rank
                    // should be good enough
                    averageTier: summoner.tier,
                    averageDivision: summoner.rank,
                  },
                });
              }

              // Update the summoner's matchesFetchedAt timestamp
              await prisma.summoner.update({
                where: {
                  id: summoner.id,
                },
                data: {
                  matchesFetchedAt: new Date(),
                },
              });

              console.log(
                `Fetched match IDs for summoner ${summoner.summonerId}`
              );
            } catch (error) {
              console.error(
                `Error fetching match IDs for summoner ${summoner.summonerId}:`,
                error
              );
            }
          })
        )
      );
    }
  } catch (error) {
    console.error("Error in collectMatchIds:", error);
  } finally {
    await prisma.$disconnect();
  }
}

collectMatchIds();
