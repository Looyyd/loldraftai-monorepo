import React, { useState, useEffect, useMemo } from "react";
import { Loader2 } from "lucide-react";
import type { Champion, Team, SelectedSpot, FavoriteChampions, Elo } from "@draftking/ui/lib/types";
import { championIndexToFavoritesPosition } from "@draftking/ui/lib/types";

interface ChampionWinrate {
  champion: Champion;
  winrate: number;
}

interface BestChampionSuggestionProps {
  team1: Team;
  team2: Team;
  selectedSpot: SelectedSpot;
  favorites: FavoriteChampions;
  remainingChampions: Champion[];
  elo: Elo;
  patch: string;
  baseApiUrl: string;
  WinrateBar: React.ComponentType<{ team1Winrate: number }>;
}

// Helper functions from DraftAnalysis
const formatTeamData = (team: Team): (number | "UNKNOWN")[] => {
  const championsIds: (number | "UNKNOWN")[] = [];
  for (let i = 0; i < 5; i++) {
    championsIds.push(team[i as keyof Team]?.id ?? "UNKNOWN");
  }
  return championsIds;
};

const eloToNumerical = (elo: Elo): number => {
  const elos = ["emerald", "low diamond", "high diamond", "master +"] as const;
  return elos.indexOf(elo);
};

export const BestChampionSuggestion = ({
  team1,
  team2,
  selectedSpot,
  favorites,
  remainingChampions,
  elo,
  patch,
  baseApiUrl,
  WinrateBar,
}: BestChampionSuggestionProps) => {
  const [championData, setChampionData] = useState<ChampionWinrate[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const championsIdsToConsider = useMemo(() => {
    const favoritesForSpot =
      favorites[championIndexToFavoritesPosition(selectedSpot.championIndex)];
    const remainingChampionsIds = remainingChampions.map(
      (champion) => champion.id
    );
    return favoritesForSpot.filter((favorite) =>
      remainingChampionsIds.includes(favorite)
    );
  }, [selectedSpot, favorites, remainingChampions]);

  useEffect(() => {
    const findBestChampion = async () => {
      setLoading(true);
      try {
        const predictionPromises = championsIdsToConsider.map(async (champId) => {
          const newTeam =
            selectedSpot.teamIndex === 1 ? { ...team1 } : { ...team2 };
          const champion = remainingChampions.find((c) => c.id === champId);

          if (!champion) {
            console.error(
              "Champion in favorites not found in champions list:",
              champId
            );
            return null;
          }

          newTeam[selectedSpot.championIndex] = champion;

          const requestBody = {
            champion_ids: [
              ...formatTeamData(
                selectedSpot.teamIndex === 1 ? newTeam : team1
              ),
              ...formatTeamData(
                selectedSpot.teamIndex === 2 ? newTeam : team2
              ),
            ],
            numerical_elo: eloToNumerical(elo),
            patch,
          };

          const response = await fetch(`${baseApiUrl}/api/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestBody),
          });

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const result = await response.json();
          const winrate = result.win_probability * 100;

          return {
            champion,
            winrate:
              selectedSpot.teamIndex === 1 ? winrate : 100 - winrate,
          };
        });

        const results = await Promise.all(predictionPromises);
        const championsWinrates = results
          .filter((result): result is ChampionWinrate => result !== null)
          .sort((a, b) => b.winrate - a.winrate);

        setChampionData(championsWinrates);
      } catch (error) {
        console.error("Error in finding best champion:", error);
        setError("Failed to load champion suggestions. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    void findBestChampion();
  }, [
    selectedSpot,
    team1,
    team2,
    championsIdsToConsider,
    elo,
    patch,
    baseApiUrl,
    remainingChampions,
  ]);

  useEffect(() => {
    if (loading) {
      window.scrollTo({
        top: document.body.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [loading]);

  return (
    <div className="mt-5 rounded border p-4">
      <div>
        <h6 className="mb-2 text-lg font-semibold">
          Champion Suggestions (Best First)
        </h6>
        <div>
          {loading ? (
            <div className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <p>Loading best champion suggestion...</p>
            </div>
          ) : error ? (
            <p className="text-red-500">{error}</p>
          ) : (
            championData.map((championWinrate, index) => (
              <div key={index} className="mb-2 flex flex-col items-center">
                <h6 className="text-lg font-semibold">{`${
                  championWinrate.champion.name
                }: ${championWinrate.winrate.toFixed(1)}%`}</h6>
                <WinrateBar
                  team1Winrate={
                    selectedSpot.teamIndex === 1
                      ? championWinrate.winrate
                      : 100 - championWinrate.winrate
                  }
                />
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}; 