import { AnalysisParent as SharedAnalysisParent } from "@draftking/ui/components/draftking/AnalysisParent";
import type {
  Champion,
  Team,
  FavoriteChampions,
  SelectedSpot,
} from "@draftking/ui/lib/types";
import { DraftAnalysis } from "./DraftAnalysis";
import { BestChampionSuggestion } from "./BestChampionSuggestion";
import { useDraftStore } from "@/app/stores/draftStore";
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
}

const AnalysisParent = (props: AnalysisParentProps) => {
  const { currentPatch, patches, setCurrentPatch, setPatchList } =
    useDraftStore();
  const [elo, setElo] = usePersistedElo();

  return (
    <SharedAnalysisParent
      {...props}
      currentPatch={currentPatch}
      patches={patches}
      setCurrentPatch={setCurrentPatch}
      setPatchList={setPatchList}
      DraftAnalysis={DraftAnalysis}
      BestChampionSuggestion={BestChampionSuggestion}
      elo={elo}
      setElo={setElo}
      baseApiUrl={
        process.env.NEXT_PUBLIC_API_BASE_URL ?? "https://loldraftai.com"
      }
    />
  );
};

export default AnalysisParent;
