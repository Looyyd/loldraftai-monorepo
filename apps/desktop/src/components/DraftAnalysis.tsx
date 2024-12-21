import { DraftAnalysis as SharedDraftAnalysis } from "@draftking/ui/components/draftking/DraftAnalysis";
import type { Team, Elo } from "@draftking/ui/lib/types";
import { DraftAnalysisShowcase } from "./DraftAnalysisShowcase";

interface DraftAnalysisProps {
  team1: Team;
  team2: Team;
  elo: Elo;
  patch: string;
}

export const DraftAnalysis = (props: DraftAnalysisProps) => {
  return (
    <SharedDraftAnalysis
      {...props}
      baseApiUrl="http://localhost:3000"
      DraftAnalysisShowcase={DraftAnalysisShowcase}
    />
  );
}; 