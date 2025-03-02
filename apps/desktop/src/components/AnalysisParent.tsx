import { AnalysisParent as SharedAnalysisParent } from "@draftking/ui/components/draftking/AnalysisParent";
import type {
  Champion,
  Team,
  SelectedSpot,
  FavoriteChampions,
} from "@draftking/ui/lib/types";
import { VERCEL_URL } from "../utils";
import { DraftAnalysis } from "./DraftAnalysis";
import { BestChampionSuggestion } from "./BestChampionSuggestion";
import { usePersistedElo } from "@draftking/ui/hooks/usePersistedState";

interface AnalysisParentProps {
  team1: Team;
  team2: Team;
  selectedSpot: SelectedSpot | null;
  setSelectedSpot: (spot: SelectedSpot | null) => void;
  favorites: FavoriteChampions;
  remainingChampions: Champion[];
  analysisTrigger: number;
  resetAnalysisTrigger?: number;
  currentPatch: string;
  patches: string[];
  setCurrentPatch: (patch: string) => void;
  setPatchList: (patches: string[]) => void;
}

export const AnalysisParent = (props: AnalysisParentProps) => {
  const [elo, setElo] = usePersistedElo();

  return (
    <SharedAnalysisParent
      {...props}
      elo={elo}
      setElo={setElo}
      baseApiUrl={VERCEL_URL}
      DraftAnalysis={DraftAnalysis}
      BestChampionSuggestion={BestChampionSuggestion}
    />
  );
};
