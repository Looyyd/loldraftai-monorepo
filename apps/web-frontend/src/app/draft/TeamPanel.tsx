import { TeamPanel as SharedTeamPanel } from "@draftking/ui/components/draftking/TeamPanel";
import type {
  Team,
  TeamIndex,
  ChampionIndex,
  SelectedSpot,
} from "@draftking/ui/lib/types";
import type { ImageComponent } from "@draftking/ui/lib/types";
import CloudFlareImage from "@/components/CloudFlareImage";

interface TeamPanelProps {
  team: Team;
  is_first_team: boolean;
  onDeleteChampion: (index: ChampionIndex) => void;
  selectedSpot: SelectedSpot | null;
  onSpotSelected: (index: ChampionIndex, team: TeamIndex) => void;
  setTeam: (team: Team) => void;
}

const TeamPanel: React.FC<TeamPanelProps> = (props) => {
  return (
    <SharedTeamPanel
      {...props}
      ImageComponent={CloudFlareImage as ImageComponent}
    />
  );
};

export default TeamPanel;
