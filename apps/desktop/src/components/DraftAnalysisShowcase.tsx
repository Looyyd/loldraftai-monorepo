import { DraftAnalysisShowcase as SharedDraftAnalysisShowcase } from "@draftking/ui/components/draftking/DraftAnalysisShowcase";
import type { Team } from "@draftking/ui/lib/types";
import { WinrateBar } from "./WinrateBar";

interface DraftAnalysisShowcaseProps {
  prediction: {
    win_probability: number;
    gold_diff_15min: number[];
    champion_impact: number[];
  };
  team1: Team;
  team2: Team;
}

export const DraftAnalysisShowcase = (props: DraftAnalysisShowcaseProps) => {
  return <SharedDraftAnalysisShowcase {...props} WinrateBar={WinrateBar} />;
}; 