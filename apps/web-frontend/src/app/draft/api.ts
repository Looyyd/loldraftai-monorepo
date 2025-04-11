import type {
  Team,
  Elo,
  ChampionIndex,
  DetailedPrediction,
} from "@draftking/ui/lib/types";
import { eloToNumerical } from "@draftking/ui/lib/draftLogic";

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

interface ModelMetadata {
  patches: string[];
  last_modified: string;
}

export const getModelMetadata = async (): Promise<ModelMetadata> => {
  const response = await fetch("/api/metadata");
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
};

export const predictGame = async (
  team1: Team,
  team2: Team,
  elo: Elo,
  patch?: string
): Promise<Prediction> => {
  const requestBody = {
    champion_ids: [...formatTeamData(team1), ...formatTeamData(team2)],
    numerical_elo: eloToNumerical(elo),
    patch,
  };

  console.log("Request body:", requestBody);
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

export const predictGameInDepth = async (
  team1: Team,
  team2: Team,
  elo: Elo,
  patch?: string
): Promise<DetailedPrediction> => {
  const requestBody = {
    champion_ids: [...formatTeamData(team1), ...formatTeamData(team2)],
    numerical_elo: eloToNumerical(elo),
    patch,
  };

  console.log("Request body:", requestBody);

  const response = await fetch("/api/predict-in-depth", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error("Error response:", errorText);
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = (await response.json()) as DetailedPrediction;
  return {
    ...data,
    win_probability: data.win_probability * 100,
  };
};
