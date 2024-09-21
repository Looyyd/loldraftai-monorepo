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
  // Short-term limit: 30 requests every 10 seconds
  minTime: 40, // 40ms between requests (25 requests per second)


  // Limit: 1600 requests every 1 minute
  reservoir: 1600,
  reservoirRefreshAmount: 1600,
  reservoirRefreshInterval: 60 * 1000, // 1 minute

  // Adjust maxConcurrent based on your needs and system capabilities
  maxConcurrent: 30,
});

async function fetchPuuids() {
  try {
    while (true) {
      // Fetch summoners without a PUUID
      const summoners = await prisma.summoner.findMany({
        where: {
          puuid: null,
          region: region,
        },
        take: 1000, // Adjust the batch size as needed
      });

      if (summoners.length === 0) {
        console.log("No summoners without PUUID found. Sleeping for 1 minute.");
        await sleep(60 * 1000);
        continue;
      }

      console.log(`Found ${summoners.length} summoners without PUUID.`);

      await Promise.all(
        summoners.map((summoner) =>
          limiter.schedule(async () => {
            try {
              const summonerDTO = await riotApiClient.getSummonerById(
                summoner.summonerId
              );

              // Update the summoner record with the fetched PUUID
              await prisma.summoner.update({
                where: {
                  id: summoner.id,
                },
                data: {
                  puuid: summonerDTO.puuid,
                },
              });

              console.log(`Updated PUUID for summoner ${summoner.summonerId}`);
            } catch (error) {
              console.error(
                `Error updating summoner ${summoner.summonerId}:`,
                error
              );
            }
          })
        )
      );
    }
  } catch (error) {
    console.error("Error in convertSummonerIdsToPuuids:", error);
  } finally {
    await prisma.$disconnect();
  }
}

fetchPuuids();
