import React, { useEffect, useMemo, useState } from "react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Button } from "../ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";
import type {
  Champion,
  Team,
  FavoriteChampions,
  SelectedSpot,
  Elo,
  SuggestionMode,
} from "@draftking/ui/lib/types";
import { elos } from "@draftking/ui/lib/types";
import { championIndexToFavoritesPosition } from "@draftking/ui/lib/types";
import { SparklesIcon, LightBulbIcon } from "@heroicons/react/24/solid";
import { LowPickrateWarning } from "./LowPickrateWarning";
import { HelpCircle } from "lucide-react";
import { usePersistedState } from "../../hooks/usePersistedState";

interface AnalysisParentProps {
  team1: Team;
  team2: Team;
  selectedSpot: SelectedSpot | null;
  setSelectedSpot: (spot: SelectedSpot | null) => void;
  favorites: FavoriteChampions;
  remainingChampions: Champion[];
  analysisTrigger: number;
  resetAnalysisTrigger?: number;
  elo: Elo;
  setElo: (elo: Elo) => void;
  currentPatch: string;
  patches: string[];
  setCurrentPatch: (patch: string) => void;
  baseApiUrl: string;
  DraftAnalysis: React.ComponentType<{
    team1: Team;
    team2: Team;
    elo: Elo;
    patch: string;
  }>;
  BestChampionSuggestion: React.ComponentType<{
    team1: Team;
    team2: Team;
    selectedSpot: SelectedSpot;
    favorites: FavoriteChampions;
    remainingChampions: Champion[];
    elo: Elo;
    patch: string;
    suggestionMode: SuggestionMode;
  }>;
  setPatchList: (patches: string[]) => void;
}

interface EloSelectProps {
  elo: Elo;
  setElo: (elo: Elo) => void;
}

const EloSelect = ({ elo, setElo }: EloSelectProps) => {
  return (
    <Select value={elo} onValueChange={setElo}>
      <SelectTrigger className="w-[140px]">
        <SelectValue placeholder="Select Elo" />
      </SelectTrigger>
      <SelectContent>
        <SelectGroup>
          {elos.map((eloOption) => (
            <SelectItem key={eloOption} value={eloOption}>
              {eloOption.toUpperCase()}
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  );
};

interface PatchSelectProps {
  currentPatch: string;
  patches: string[];
  setCurrentPatch: (patch: string) => void;
}

const PatchSelect = ({
  currentPatch,
  patches,
  setCurrentPatch,
}: PatchSelectProps) => (
  <Select value={currentPatch} onValueChange={setCurrentPatch}>
    <SelectTrigger className="w-[110px]">
      <SelectValue placeholder="Select Patch" />
    </SelectTrigger>
    <SelectContent>
      <SelectGroup>
        {patches.reverse().map((patchOption) => (
          <SelectItem key={patchOption} value={patchOption}>
            {patchOption}
          </SelectItem>
        ))}
      </SelectGroup>
    </SelectContent>
  </Select>
);

interface ModelMetadata {
  patches: string[];
  last_modified: string;
}

export const AnalysisParent = ({
  team1,
  team2,
  selectedSpot,
  setSelectedSpot,
  favorites,
  remainingChampions,
  analysisTrigger,
  resetAnalysisTrigger = 0,
  elo,
  setElo,
  currentPatch,
  patches,
  setCurrentPatch,
  DraftAnalysis,
  BestChampionSuggestion,
  setPatchList,
  baseApiUrl,
}: AnalysisParentProps) => {
  const [showChampionSuggestion, setShowChampionSuggestion] = useState(false);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [suggestionMode, setSuggestionMode] = usePersistedState<SuggestionMode>(
    "draftking-suggestion-mode",
    "meta"
  );

  const [suggestionSelectedSpot, setSuggestionSelectedSpot] =
    useState<SelectedSpot | null>(null);

  /*
  Trying without reset on team change
  useEffect(() => {
    setShowAnalysis(false);
  }, [team1, team2, elo]);
  */

  useEffect(() => {
    if (resetAnalysisTrigger > 0) {
      setShowChampionSuggestion(false);
      setShowAnalysis(false);
      setSuggestionSelectedSpot(null);
    }
  }, [resetAnalysisTrigger]);

  useEffect(() => {
    if (analysisTrigger > 0) {
      setShowAnalysis(true);
    }
  }, [analysisTrigger]);

  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        const response = await fetch(`${baseApiUrl}/api/metadata`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const metadata: ModelMetadata = await response.json();
        setPatchList(metadata.patches);
      } catch (error) {
        console.error("Failed to fetch patches:", error);
      }
    };
    void fetchMetadata();
  }, [baseApiUrl, setPatchList]);

  const enableChampionSuggestion = useMemo(() => {
    if (!selectedSpot) return false;

    if (suggestionMode === "favorites") {
      return (
        favorites[championIndexToFavoritesPosition(selectedSpot.championIndex)]
          .length > 0
      );
    }

    return true;
  }, [favorites, suggestionMode, selectedSpot]);

  const isSameSpot = (
    spot1: SelectedSpot | null,
    spot2: SelectedSpot | null
  ): boolean => {
    if (!spot1 || !spot2) return false;
    return (
      spot1.teamIndex === spot2.teamIndex &&
      spot1.championIndex === spot2.championIndex
    );
  };

  // Helper functions to make logic more readable
  const isSuggestingForCurrentSpot = useMemo((): boolean => {
    return (
      showChampionSuggestion &&
      (selectedSpot === null ||
        isSameSpot(selectedSpot, suggestionSelectedSpot))
    );
  }, [showChampionSuggestion, selectedSpot, suggestionSelectedSpot]);

  const canShowSuggestions = useMemo((): boolean => {
    return enableChampionSuggestion;
  }, [enableChampionSuggestion]);

  const handleSuggestionButtonClick = (): void => {
    if (isSuggestingForCurrentSpot) {
      // Hide suggestions
      setShowChampionSuggestion(false);
      setSuggestionSelectedSpot(null);
    } else {
      // Show suggestions
      setShowChampionSuggestion(true);
      setSuggestionSelectedSpot(selectedSpot);
      setSelectedSpot(null);
    }
  };

  const getSuggestionButtonTooltip = (): string => {
    if (isSuggestingForCurrentSpot) {
      return "Click to hide champion suggestions.";
    } else if (!selectedSpot) {
      return "Click on a team position to select it.";
    } else if (suggestionMode === "favorites" && !enableChampionSuggestion) {
      return "Right click a champion in the list to add to favorites.";
    } else {
      return "Click to show champion suggestions.";
    }
  };

  return (
    <div className="draft-analysis p-5">
      <LowPickrateWarning
        teamOne={team1}
        teamTwo={team2}
        currentPatch={currentPatch}
      />

      <div className="flex flex-wrap items-end justify-center gap-4">
        <div className="flex flex-col">
          <span className="text-xs text-neutral-500 mb-1 ml-1">Config</span>
          <div className="flex items-center gap-2 p-2 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-900/50">
            <EloSelect elo={elo} setElo={setElo} />
            <PatchSelect
              currentPatch={currentPatch}
              patches={patches}
              setCurrentPatch={setCurrentPatch}
            />
            <div className="flex items-center gap-1">
              <Select
                value={suggestionMode}
                onValueChange={(value: SuggestionMode) =>
                  setSuggestionMode(value)
                }
              >
                <SelectTrigger className="w-[200px]">
                  <SelectValue placeholder="Suggestion mode" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="favorites">Favorites Only</SelectItem>
                  <SelectItem value="meta">Meta Champions</SelectItem>
                  <SelectItem value="all">All Champions</SelectItem>
                </SelectContent>
              </Select>
              <TooltipProvider delayDuration={0}>
                <Tooltip>
                  <TooltipTrigger>
                    <HelpCircle className="h-5 w-5 text-gray-500" />
                  </TooltipTrigger>
                  <TooltipContent className="max-w-[350px]">
                    <p className="font-semibold mb-2">Suggestion Modes:</p>
                    <ul className="list-disc pl-4 space-y-1">
                      <li>
                        <strong>Favorites Only:</strong> Only shows suggestions
                        for your favorite champions
                      </li>
                      <li>
                        <strong>Meta Champions:</strong> Shows suggestions for
                        your favorite champions and champions with &gt;0.5% pick
                        rate in this role
                      </li>
                      <li>
                        <strong>All Champions:</strong> Shows suggestions for
                        all champions. Note: Predictions for rarely-played
                        champions may be less accurate due to limited data.
                      </li>
                    </ul>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>
        </div>

        <Button
          variant="outline"
          onClick={() => setShowAnalysis(!showAnalysis)}
          className="h-[58px] text-base font-medium px-6"
        >
          {showAnalysis ? "Hide Analysis" : "Analyze Draft"}{" "}
          <SparklesIcon className="inline-block h-5 w-5 ml-1" />
        </Button>

        <TooltipProvider delayDuration={0}>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="inline-block">
                <Button
                  variant="outline"
                  onClick={handleSuggestionButtonClick}
                  disabled={!isSuggestingForCurrentSpot && !canShowSuggestions}
                  className="h-[58px] text-base font-medium px-6"
                >
                  {isSuggestingForCurrentSpot
                    ? "Hide Suggestions"
                    : "Suggest Champion"}{" "}
                  <LightBulbIcon className="inline-block h-5 w-5 ml-1" />
                </Button>
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p>{getSuggestionButtonTooltip()}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {showAnalysis && (
        <DraftAnalysis
          team1={team1}
          team2={team2}
          elo={elo}
          patch={currentPatch}
        />
      )}

      {showChampionSuggestion && suggestionSelectedSpot && (
        <BestChampionSuggestion
          team1={team1}
          team2={team2}
          selectedSpot={suggestionSelectedSpot}
          favorites={favorites}
          remainingChampions={remainingChampions}
          elo={elo}
          patch={currentPatch}
          suggestionMode={suggestionMode}
        />
      )}
    </div>
  );
};
