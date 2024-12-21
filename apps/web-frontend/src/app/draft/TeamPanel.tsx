import Image from "next/image";
import { TeamPanel as SharedTeamPanel } from "@draftking/ui/components/draftking/TeamPanel";
import type { Team, TeamIndex, ChampionIndex, SelectedSpot } from "@/app/types";

interface TeamPanelProps {
  team: Team;
  is_first_team: boolean;
  onDeleteChampion: (index: ChampionIndex) => void;
  selectedSpot: SelectedSpot | null;
  onSpotSelected: (index: ChampionIndex, team: TeamIndex) => void;
}

const TeamPanel: React.FC<TeamPanelProps> = (props) => {
  return <SharedTeamPanel {...props} ImageComponent={Image} />;
};

export default TeamPanel;
