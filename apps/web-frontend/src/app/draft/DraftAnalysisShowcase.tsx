import { DraftAnalysisShowcase as SharedDraftAnalysisShowcase } from "@draftking/ui/components/draftking/DraftAnalysisShowcase";
import type { ImageComponent, Team } from "@draftking/ui/lib/types";
import CloudFlareImage from "@/components/CloudFlareImage";
interface DraftAnalysisShowcaseProps {
  prediction: {
    win_probability: number;
    gold_diff_15min: number[];
    champion_impact: number[];
    time_bucketed_predictions: Record<string, number>;
  };
  team1: Team;
  team2: Team;
}

export const DraftAnalysisShowcase = (props: DraftAnalysisShowcaseProps) => {
  return (
    <SharedDraftAnalysisShowcase
      {...props}
      ImageComponent={CloudFlareImage as ImageComponent}
    />
  );
};
