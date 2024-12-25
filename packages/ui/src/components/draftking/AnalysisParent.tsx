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
} from "@draftking/ui/lib/types";
import { championIndexToFavoritesPosition } from "@draftking/ui/lib/types";
import { SparklesIcon, LightBulbIcon } from "@heroicons/react/24/solid";

interface AnalysisParentProps {
  team1: Team;
  team2: Team;
  selectedSpot: SelectedSpot | null;
  favorites: FavoriteChampions;
  remainingChampions: Champion[];
  analysisTrigger: number;
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
  }>;
  setPatchList: (patches: string[]) => void;
}

interface EloSelectProps {
  elo: Elo;
  setElo: (elo: Elo) => void;
}

const EloSelect = ({ elo, setElo }: EloSelectProps) => {
  const elos = ["emerald", "low diamond", "high diamond", "master +"] as const;
  
  return (
    <Select value={elo} onValueChange={setElo}>
      <SelectTrigger>
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

const PatchSelect = ({ currentPatch, patches, setCurrentPatch }: PatchSelectProps) => (
  <Select value={currentPatch} onValueChange={setCurrentPatch}>
    <SelectTrigger>
      <SelectValue placeholder="Select Patch" />
    </SelectTrigger>
    <SelectContent>
      <SelectGroup>
        {patches.map((patchOption) => (
          <SelectItem key={patchOption} value={patchOption}>
            {patchOption}
          </SelectItem>
        ))}
      </SelectGroup>
    </SelectContent>
  </Select>
);

interface AnalyzeDraftButtonProps {
  toggleAnalyzeDraft: () => void;
  showAnalysis: boolean;
}

const AnalyzeDraftButton = ({
  toggleAnalyzeDraft,
  showAnalysis,
}: AnalyzeDraftButtonProps) => (
  <Button variant="outline" onClick={toggleAnalyzeDraft}>
    {showAnalysis ? "Hide Analysis" : "Analyze Draft"}{" "}
    <SparklesIcon className="inline-block h-5 w-5 ml-1" />
  </Button>
);

interface ChampionSuggestionButtonProps {
  enableChampionSuggestion: boolean;
  toggleChampionSuggestion: () => void;
  showChampionSuggestion: boolean;
  selectedSpot: SelectedSpot | null;
}

const ChampionSuggestionButton = ({
  enableChampionSuggestion,
  toggleChampionSuggestion,
  showChampionSuggestion,
  selectedSpot,
}: ChampionSuggestionButtonProps) =>
  enableChampionSuggestion ? (
    <Button variant="outline" onClick={toggleChampionSuggestion}>
      {showChampionSuggestion ? "Hide Suggestions" : "Suggest Champion"}{" "}
      <LightBulbIcon className="inline-block h-5 w-5 ml-1" />
    </Button>
  ) : (
    <TooltipProvider delayDuration={0}>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="inline-block">
            <Button variant="outline" onClick={toggleChampionSuggestion} disabled>
              {showChampionSuggestion ? "Hide Suggestions" : "Suggest Champion"}{" "}
              <LightBulbIcon className="inline-block h-5 w-5 ml-1" />
            </Button>
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <p>
            {!selectedSpot
              ? "Click on a team position to select a position."
              : "Right click a champion in the list to add to favorites."}
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );

interface ModelMetadata {
  patches: string[];
  last_modified: string;
}

export const AnalysisParent = ({
  team1,
  team2,
  selectedSpot,
  favorites,
  remainingChampions,
  analysisTrigger,
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

  useEffect(() => {
    setShowChampionSuggestion(false);
  }, [selectedSpot, team1, team2, elo]);

  useEffect(() => {
    setShowAnalysis(false);
  }, [team1, team2, elo]);

  useEffect(() => {
    if (analysisTrigger > 0) {
      setShowAnalysis(true);
    }
  }, [analysisTrigger]);

  useEffect(() => {
    if (showAnalysis) {
      window.scrollTo({
        top: document.body.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [showAnalysis]);

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
    return (
      selectedSpot &&
      favorites[championIndexToFavoritesPosition(selectedSpot.championIndex)]
        .length > 0
    );
  }, [selectedSpot, favorites]);

  return (
    <div className="draft-analysis p-5">
      <div className="flex flex-wrap items-stretch justify-center">
        <div className="flex w-full p-1 sm:w-auto">
          <div className="flex-1">
            <EloSelect elo={elo} setElo={setElo} />
          </div>
        </div>
        <div className="flex w-full p-1 sm:w-auto">
          <div className="flex-1">
            <PatchSelect
              currentPatch={currentPatch}
              patches={patches}
              setCurrentPatch={setCurrentPatch}
            />
          </div>
        </div>
        <div className="flex w-full p-1 sm:w-auto">
          <div className="flex-1">
            <AnalyzeDraftButton
              toggleAnalyzeDraft={() => setShowAnalysis(!showAnalysis)}
              showAnalysis={showAnalysis}
            />
          </div>
        </div>
        <div className="flex w-full p-1 sm:w-auto">
          <div className="flex-1">
            <ChampionSuggestionButton
              enableChampionSuggestion={enableChampionSuggestion ?? false}
              toggleChampionSuggestion={() =>
                setShowChampionSuggestion(!showChampionSuggestion)
              }
              showChampionSuggestion={showChampionSuggestion}
              selectedSpot={selectedSpot}
            />
          </div>
        </div>
      </div>

      {showAnalysis && (
        <DraftAnalysis
          team1={team1}
          team2={team2}
          elo={elo}
          patch={currentPatch}
        />
      )}

      {showChampionSuggestion && selectedSpot && (
        <BestChampionSuggestion
          team1={team1}
          team2={team2}
          selectedSpot={selectedSpot}
          favorites={favorites}
          remainingChampions={remainingChampions}
          elo={elo}
          patch={currentPatch}
        />
      )}
    </div>
  );
}; 