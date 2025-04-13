import { DraftAnalysisShowcase as SharedDraftAnalysisShowcase } from "@draftking/ui/components/draftking/DraftAnalysisShowcase";
import type { Team, DetailedPrediction } from "@draftking/ui/lib/types";
import { PlainImage } from "./PlainImage";

interface DraftAnalysisShowcaseProps {
  prediction: DetailedPrediction;
  team1: Team;
  team2: Team;
}

export const DraftAnalysisShowcase = (props: DraftAnalysisShowcaseProps) => {
  return <SharedDraftAnalysisShowcase {...props} ImageComponent={PlainImage} />;
};
