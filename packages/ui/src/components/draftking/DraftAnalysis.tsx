import React, { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import type { Team, Elo } from "@draftking/ui/lib/types";

interface PredictionResult {
  win_probability: number;
  gold_diff_15min: number[];
  champion_impact: number[];
}

interface DraftAnalysisProps {
  team1: Team;
  team2: Team;
  elo: Elo;
  patch: string;
  baseApiUrl: string;
  DraftAnalysisShowcase: React.ComponentType<{
    prediction: PredictionResult;
    team1: Team;
    team2: Team;
  }>;
}

// Helper function to format team data for API
const formatTeamData = (team: Team): (number | "UNKNOWN")[] => {
  const championsIds: (number | "UNKNOWN")[] = [];
  for (let i = 0; i < 5; i++) {
    championsIds.push(team[i as keyof Team]?.id ?? "UNKNOWN");
  }
  return championsIds;
};

// Helper function to convert elo to numerical value
const eloToNumerical = (elo: Elo): number => {
  const elos = ["emerald", "low diamond", "high diamond", "master +"] as const;
  return elos.indexOf(elo);
};

export const DraftAnalysis = ({
  team1,
  team2,
  elo,
  patch,
  baseApiUrl,
  DraftAnalysisShowcase,
}: DraftAnalysisProps) => {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPrediction = async () => {
      setLoading(true);
      setError(null);
      try {
        const requestBody = {
          champion_ids: [...formatTeamData(team1), ...formatTeamData(team2)],
          numerical_elo: eloToNumerical(elo),
          patch,
        };

        const response = await fetch(`${baseApiUrl}/api/predict-in-depth`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        // The API returns win_probability as a decimal, convert to percentage
        setPrediction({
          ...result,
          win_probability: result.win_probability * 100,
        });
      } catch (err) {
        console.error("Error fetching prediction:", err);
        setError("Failed to load prediction. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    void fetchPrediction();
  }, [team1, team2, elo, patch, baseApiUrl]);

  return (
    <div className="mt-5 rounded border border-gray-200 p-4">
      <div className="flex items-center justify-between">
        <h6 className="text-lg font-semibold">Draft Analysis</h6>
      </div>
      <div className="mb-2.5 mt-2 grid grid-cols-1 gap-2">
        <div>
          {loading ? (
            <div className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <p>Loading...</p>
            </div>
          ) : error ? (
            <p className="text-red-500">{error}</p>
          ) : prediction ? (
            <DraftAnalysisShowcase
              prediction={prediction}
              team1={team1}
              team2={team2}
            />
          ) : (
            <p>No data available</p>
          )}
        </div>
      </div>
    </div>
  );
}; 