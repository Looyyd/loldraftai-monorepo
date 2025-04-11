import { useState, useEffect, useMemo } from "react";
import { Loader2 } from "lucide-react";
import type {
  Champion,
  Team,
  SelectedSpot,
  FavoriteChampions,
  Elo,
  ImageComponent,
  SuggestionMode,
} from "@draftking/ui/lib/types";
import { champions } from "@draftking/ui/lib/champions";
import { championIndexToFavoritesPosition } from "@draftking/ui/lib/types";
import { eloToNumerical } from "@draftking/ui/lib/draftLogic";
import { WinrateBar } from "./WinrateBar";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "../ui/tooltip";
import { AlertTriangle } from "lucide-react";
import { StarIcon } from "@heroicons/react/24/solid";

import { getChampionPlayRates } from "../../lib/champions";

interface BestChampionSuggestionProps {
  team1: Team;
  team2: Team;
  selectedSpot: SelectedSpot;
  favorites: FavoriteChampions;
  remainingChampions: Champion[];
  elo: Elo;
  patch: string;
  baseApiUrl: string;
  ImageComponent: ImageComponent;
  suggestionMode: SuggestionMode;
}

const PICKRATE_THRESHOLD = 0.3;
const META_THRESHOLD = 0.5;

const roleIndexToKey = {
  0: "TOP",
  1: "JUNGLE",
  2: "MIDDLE",
  3: "BOTTOM",
  4: "UTILITY",
} as const;

const roleToDisplayName = {
  TOP: "Toplane",
  JUNGLE: "Jungle",
  MIDDLE: "Midlane",
  BOTTOM: "Botlane",
  UTILITY: "Support",
} as const;

export const BestChampionSuggestion = ({
  team1,
  team2,
  selectedSpot,
  favorites,
  remainingChampions,
  elo,
  patch,
  baseApiUrl,
  ImageComponent,
  suggestionMode,
}: BestChampionSuggestionProps) => {
  const [championWinrates, setChampionWinrates] = useState<
    Array<{ champion: Champion; winrate: number }>
  >([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const championsIdsToConsider = useMemo(() => {
    const favoritesForSpot =
      favorites[championIndexToFavoritesPosition(selectedSpot.championIndex)];

    if (suggestionMode === "favorites") return favoritesForSpot;

    if (suggestionMode === "meta") {
      // Get meta champions based on play rate
      const metaChampions = champions
        .filter((champion) => {
          const roleKey = roleIndexToKey[selectedSpot.championIndex];
          const playRates = getChampionPlayRates(champion.id, patch);
          return playRates && playRates[roleKey] >= META_THRESHOLD;
        })
        .map((c) => c.id);

      // Combine meta champions with favorites and remove duplicates
      return [...new Set([...metaChampions, ...favoritesForSpot])];
    }

    // "all" mode
    return champions.map((c) => c.id);
  }, [suggestionMode, selectedSpot, favorites, patch]);

  const championData = useMemo(() => {
    return championWinrates.map(({ champion, winrate }) => ({
      champion,
      winrate,
      isAvailable: remainingChampions.some((c) => c.id === champion.id),
    }));
  }, [championWinrates, remainingChampions]);

  useEffect(() => {
    const fetchWinrates = async () => {
      setLoading(true);
      try {
        const response = await fetch(`${baseApiUrl}/api/champion-suggestions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            team1,
            team2,
            championIds: championsIdsToConsider,
            selectedTeamIndex: selectedSpot.teamIndex,
            selectedChampionIndex: selectedSpot.championIndex,
            elo: eloToNumerical(elo),
            patch,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const predictions: { championId: number; winrate: number }[] =
          await response.json();
        const winrates = predictions
          .map((pred: { championId: number; winrate: number }) => ({
            champion: champions.find((c) => c.id === pred.championId)!,
            winrate: pred.winrate,
          }))
          .sort((a, b) => b.winrate - a.winrate);

        setChampionWinrates(winrates);
      } catch (error) {
        console.error("Error in finding best champion:", error);
        setError("Failed to load champion suggestions. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    void fetchWinrates();
  }, [
    selectedSpot,
    team1,
    team2,
    championsIdsToConsider,
    elo,
    patch,
    baseApiUrl,
  ]);

  const isLowPickrate = (champion: Champion) => {
    const roleKey = roleIndexToKey[selectedSpot.championIndex];
    const playRates = getChampionPlayRates(champion.id, patch);
    return playRates ? playRates[roleKey] < PICKRATE_THRESHOLD : false;
  };

  const isFavorite = (champion: Champion) => {
    const favoritesForSpot =
      favorites[championIndexToFavoritesPosition(selectedSpot.championIndex)];
    return favoritesForSpot.includes(champion.id);
  };

  return (
    <div className="mt-5 rounded-lg border border-gray-200 bg-gradient-to-b from-white to-gray-50 dark:from-gray-900 dark:to-gray-950 p-6 shadow-sm">
      <div>
        <h6 className="text-xl font-semibold  mb-4">
          Champion Suggestions for{" "}
          <span
            className={
              selectedSpot.teamIndex === 1 ? "text-blue-500" : "text-red-500"
            }
          >
            {selectedSpot.teamIndex === 1 ? "BLUE" : "RED"}
          </span>{" "}
          {roleToDisplayName[roleIndexToKey[selectedSpot.championIndex]]}
        </h6>

        <div className="space-y-1">
          {loading ? (
            <div className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
              <p>Loading best champion suggestion...</p>
            </div>
          ) : error ? (
            <p className="text-red-500">{error}</p>
          ) : (
            championData.map((championWinrate, index) => (
              <div
                key={index}
                className={`relative rounded-md p-3 ${
                  !championWinrate.isAvailable ? "opacity-50" : ""
                }`}
              >
                <div className="flex flex-col items-center gap-2">
                  <div className="flex items-center gap-3 mb-1">
                    <div className="relative">
                      <div className="absolute -left-2 -top-2 flex h-5 w-5 items-center justify-center rounded-full bg-slate-700 text-xs font-bold text-white">
                        {index + 1}
                      </div>

                      <div className="absolute -right-2 -top-2 flex gap-1">
                        {isFavorite(championWinrate.champion) && (
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger>
                                <StarIcon
                                  className="h-4 w-4 text-yellow-500"
                                  stroke="black"
                                  strokeWidth={2}
                                />
                              </TooltipTrigger>
                              <TooltipContent>Favorite Champion</TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        )}
                        {isLowPickrate(championWinrate.champion) && (
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger>
                                <AlertTriangle className="h-4 w-4 text-amber-500" />
                              </TooltipTrigger>
                              <TooltipContent>
                                Uncommon pick in this role. Predictions may have
                                lower confidence.
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        )}
                      </div>

                      <ImageComponent
                        src={`/icons/champions/${championWinrate.champion.icon}`}
                        alt={championWinrate.champion.name}
                        width={40}
                        height={40}
                        className={`rounded-full ${
                          !championWinrate.isAvailable ? "grayscale" : ""
                        }`}
                      />
                    </div>
                    <h6
                      className={`text-lg font-semibold ${
                        !championWinrate.isAvailable ? "line-through" : ""
                      }`}
                    >
                      {championWinrate.champion.name}
                      <span
                        className={`ml-2 ${
                          selectedSpot.teamIndex === 1
                            ? "text-blue-500"
                            : "text-red-500"
                        }`}
                      >
                        {championWinrate.winrate.toFixed(1)}%
                      </span>
                    </h6>
                  </div>
                  <div className="w-full">
                    <WinrateBar
                      team1Winrate={
                        selectedSpot.teamIndex === 1
                          ? championWinrate.winrate
                          : 100 - championWinrate.winrate
                      }
                    />
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
