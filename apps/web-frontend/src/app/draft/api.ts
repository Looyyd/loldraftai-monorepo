import type { Team, Elo, ChampionIndex } from "@/app/types";
import { eloToNumerical } from "@/app/types";

export const formatTeamData = (team: Team) => {
  // Transform team data into the required format for the API
  const championsIds: (number | "UNKNOWN")[] = [];
  for (let i = 0; i < 5; i++) {
    championsIds.push(team[i as ChampionIndex]?.id ?? "UNKNOWN");
  }
  return championsIds;
};

interface Prediction {
  win_probability: number;
}

export const predictGame = async (
  team1: Team,
  team2: Team,
  elo: Elo
): Promise<Prediction> => {
  const requestBody = {
    champion_ids: [...formatTeamData(team1), ...formatTeamData(team2)],
    numerical_elo: eloToNumerical(elo),
  };

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = (await response.json()) as Prediction;
    return { win_probability: data.win_probability * 100 };
  } catch (error) {
    console.error("There was a problem with the prediction:", error);
    throw error;
  }
};

interface DetailedPrediction {
  win_probability: number;
  gold_diff_15min: number[];
  champion_impact: number[];
}

export const predictGameInDepth = async (
  team1: Team,
  team2: Team,
  elo: Elo
): Promise<DetailedPrediction> => {
  const requestBody = {
    champion_ids: [...formatTeamData(team1), ...formatTeamData(team2)],
    numerical_elo: eloToNumerical(elo),
  };

  const response = await fetch("/api/predict-in-depth", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = (await response.json()) as DetailedPrediction;
  return {
    ...data,
    win_probability: data.win_probability * 100,
  };
};
