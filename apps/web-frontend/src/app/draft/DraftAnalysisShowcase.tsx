import { DraftAnalysisShowcase as SharedDraftAnalysisShowcase } from "@draftking/ui/components/draftking/DraftAnalysisShowcase";
import type {
  ImageComponent,
  Team,
  DetailedPrediction,
} from "@draftking/ui/lib/types";
import CloudFlareImage from "@/components/CloudFlareImage";
interface DraftAnalysisShowcaseProps {
  prediction: DetailedPrediction;
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
