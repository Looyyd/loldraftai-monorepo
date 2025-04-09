import React, { useEffect, useState } from "react";
import { Loader2, HelpCircle } from "lucide-react";
import type { Team, Elo } from "@draftking/ui/lib/types";
import { eloToNumerical } from "@draftking/ui/lib/draftLogic";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "../ui/tooltip";

interface PredictionResult {
  win_probability: number;
  gold_diff_15min: number[];
  champion_impact: number[];
  time_bucketed_predictions: Record<string, number>;
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
    <div className="mt-5 rounded-lg border border-gray-200 bg-gradient-to-b from-white to-gray-50 dark:from-gray-900 dark:to-gray-950 p-6 shadow-sm">
      <div className="flex items-center gap-2">
        <h6 className="text-xl font-semibold brand-text">
          LoLDraftAI Analysis
        </h6>
        <TooltipProvider delayDuration={0}>
          <Tooltip>
            <TooltipTrigger>
              <HelpCircle className="h-5 w-5 text-gray-500" />
            </TooltipTrigger>
            <TooltipContent className="max-w-[350px] whitespace-normal">
              <p className="mb-2">
                <span className="brand-text">LoLDraftAI</span> analyzes drafts
                by understanding complex game dynamics and champion
                interactions, not just statistics.
              </p>
              <p className="mb-2">Understanding the analysis:</p>
              <ul className="list-disc pl-4 space-y-1">
                <li>
                  <strong>Impact Score:</strong> Shows how a champion changes
                  your team's win chance. Calculated as the difference between
                  the win rate with the champion and without them.
                  <div className="mt-2 text-sm text-blue-600 dark:text-blue-400 font-medium">
                    Pro tip: Focus your gameplay around champions with high
                    impact scores - they are your win conditions!
                  </div>
                </li>
                <li>
                  <strong>Gold@15:</strong> Predicted gold differences at 15
                  minutes.
                  <div className="mt-2 text-sm text-blue-600 dark:text-blue-400 font-medium">
                    Pro tip: Use these predictions to identify which lanes are
                    likely to have an advantage.
                  </div>
                </li>
              </ul>
              <p className="mt-2 text-sm ">
                Remember that these predictions are based on average game
                patterns - use them to create your game plan but adapt to the
                actual game state.
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      <div className="mb-2.5 mt-4 grid grid-cols-1 gap-2">
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
