import { AnalysisParent as SharedAnalysisParent } from "@draftking/ui/components/draftking/AnalysisParent";
import type {
  Champion,
  Team,
  FavoriteChampions,
  SelectedSpot,
  Elo,
} from "@/app/types";
import { DraftAnalysis } from "./DraftAnalysis";
import { BestChampionSuggestion } from "./BestChampionSuggestion";
import { useDraftStore } from "@/app/stores/draftStore";
import { useState } from "react";

interface AnalysisParentProps {
  team1: Team;
  team2: Team;
  selectedSpot: SelectedSpot | null;
  favorites: FavoriteChampions;
  remainingChampions: Champion[];
  analysisTrigger: number;
}

const AnalysisParent = (props: AnalysisParentProps) => {
  const { currentPatch, patches, setCurrentPatch, setPatchList } = useDraftStore();
  const [elo, setElo] = useState<Elo>("emerald");

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
      baseApiUrl=""
    />
  );
};

export default AnalysisParent;
