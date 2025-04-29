import { NextResponse } from "next/server";
import { Champion, Team } from "@draftking/ui/lib/types";
import { RateLimit } from "@/app/lib/rate-limit";
import { PredictionTracking } from "@/app/lib/prediction-tracking";
interface ChampionSuggestionRequest {
  team1: Team;
  team2: Team;
  championIds: number[];
  selectedTeamIndex: 1 | 2;
  selectedChampionIndex: number;
  elo: number;
  patch: string;
}

const backendUrl = process.env.INFERENCE_BACKEND_URL ?? "http://127.0.0.1:8000";
const proBackendUrl = process.env.PRO_INFERENCE_BACKEND_URL;
const backendApiKey = process.env.INFERENCE_BACKEND_API_KEY;

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    },
  });
}

export async function POST(request: Request) {
  const isAllowed = await RateLimit.checkRateLimit();
  if (!isAllowed) {
    return new NextResponse(
      JSON.stringify({ error: "Too many requests. Please try again later." }),
      {
        status: 429,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
          "Retry-After": "60",
        },
      }
    );
  }

  try {
    // Track prediction attempt (only tracks once per day per IP)
    await PredictionTracking.trackPredictionIfNeeded();

    const body: ChampionSuggestionRequest = await request.json();
    const {
      team1,
      team2,
      championIds,
      selectedTeamIndex,
      selectedChampionIndex,
      elo,
      patch,
    } = body;

    // Check if using pro model (elo = -1)
    const isPro = elo === -1;
    if (isPro && !proBackendUrl) {
      return NextResponse.json(
        { error: "Pro model not available" },
        {
          status: 404,
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Content-Type": "application/json",
          },
        }
      );
    }

    // Use pro backend URL if elo is -1 and proBackendUrl is available
    const targetUrl = isPro ? proBackendUrl! : backendUrl;

    // Prepare batch of predictions - one for each candidate champion
    const batchInputs = championIds.map((champId) => {
      const newTeam = selectedTeamIndex === 1 ? { ...team1 } : { ...team2 };
      newTeam[selectedChampionIndex as keyof Team] = {
        id: champId,
      } as Champion;

      return {
        champion_ids: [
          ...formatTeamData(selectedTeamIndex === 1 ? newTeam : team1),
          ...formatTeamData(selectedTeamIndex === 2 ? newTeam : team2),
        ],
        numerical_elo: isPro ? 0 : elo, // always 0 for pro TODO: this could be cleaner
        patch,
      };
    });

    // Make single batched request
    const response = await fetch(`${targetUrl}/predict-batch`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": backendApiKey || "",
      },
      body: JSON.stringify(batchInputs),
    });

    if (!response.ok) {
      if (response.status === 403) {
        console.error("API key validation failed");
        throw new Error("API key validation failed");
      }
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const results = await response.json();

    // Transform results to match the expected format
    const predictions = results.map(
      (result: { win_probability: number }, index: number) => ({
        championId: championIds[index],
        winrate:
          selectedTeamIndex === 1
            ? result.win_probability * 100
            : (1 - result.win_probability) * 100,
      })
    );

    return NextResponse.json(predictions, {
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "application/json",
      },
    });
  } catch (error) {
    console.error("Champion suggestions error:", error);
    return NextResponse.json(
      { error: "Failed to get champion suggestions" },
      {
        status: 500,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Content-Type": "application/json",
        },
      }
    );
  }
}

// Helper function
const formatTeamData = (team: Team): (number | "UNKNOWN")[] => {
  const championsIds: (number | "UNKNOWN")[] = [];
  for (let i = 0; i < 5; i++) {
    championsIds.push(team[i as keyof Team]?.id ?? "UNKNOWN");
  }
  return championsIds;
};
