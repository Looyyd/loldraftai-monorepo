import { AnalysisParent as SharedAnalysisParent } from "@draftking/ui/components/draftking/AnalysisParent";
import type {
  Champion,
  Team,
  SelectedSpot,
  FavoriteChampions,
  Elo,
} from "@draftking/ui/lib/types";
import { DraftAnalysis } from "./DraftAnalysis";
import { BestChampionSuggestion } from "./BestChampionSuggestion";

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
  setPatchList: (patches: string[]) => void;
}

export const AnalysisParent = (props: AnalysisParentProps) => {
  return (
    <SharedAnalysisParent
      {...props}
      baseApiUrl="http://localhost:3000"
      DraftAnalysis={DraftAnalysis}
      BestChampionSuggestion={BestChampionSuggestion}
    />
  );
};
