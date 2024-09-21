import { RiotAPIClient } from "@draftking/riot-api";
import { processMatchData } from "./utils/matchProcessing";
import { config } from "dotenv";

config();

const apiKey = process.env.X_RIOT_API_KEY;

if (!apiKey) {
  throw new Error("X_RIOT_API_KEY is not set");
}

const client = new RiotAPIClient(apiKey);

await client.getLeagueEntries("RANKED_SOLO_5x5", ["DIAMOND", "I"], 1);

const response = await client.getLeagueEntries(
  "RANKED_SOLO_5x5",
  ["CHALLENGER", "I"],
  1
);

if (!response[0]) {
  throw new Error("No summoner found");
}

const summoner = await client.getSummonerById(response[0].summonerId);

const matchIds = await client.getMatchIdsByPuuid(summoner.puuid, {
  type: "ranked",
  // 420 is ranked solo/duo queue
  // source: https://static.developer.riotgames.com/docs/lol/queues.json
  queue: 420,
});

if (!matchIds[0]) {
  throw new Error("No match ids found");
}

const processedData = await processMatchData(client, matchIds[0]);

console.log(JSON.stringify(processedData, null, 2));
