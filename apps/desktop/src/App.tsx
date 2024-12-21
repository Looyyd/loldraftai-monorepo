import { useState } from "react";
import { Button } from "@draftking/ui/components/ui/button";
import { TeamPanel } from "@draftking/ui/components/draftking/TeamPanel";
import { ChampionGrid } from "@draftking/ui/components/draftking/ChampionGrid";
import { champions } from "@draftking/ui/lib/champions";
import type { FavoriteChampions } from "@draftking/ui/components/draftking/ChampionGrid";
import type {
  Team,
  SelectedSpot,
  ChampionIndex,
  TeamIndex,
  Champion,
} from "@draftking/ui/components/draftking/TeamPanel";

// Plain image component for Electron
const PlainImage: React.FC<{
  src: string;
  alt: string;
  width: number;
  height: number;
  className?: string;
}> = (props) => <img {...props} />;

const emptyTeam: Team = [undefined, undefined, undefined, undefined, undefined];

function App() {
  const [team, setTeam] = useState<Team>(emptyTeam);
  const [selectedSpot, setSelectedSpot] = useState<SelectedSpot | null>(null);
  const [favorites, setFavorites] = useState<FavoriteChampions>({
    top: [],
    jungle: [],
    mid: [],
    bot: [],
    support: [],
  });

  const handleDeleteChampion = (index: ChampionIndex) => {
    const newTeam = [...team];
    newTeam[index] = undefined;
    setTeam(newTeam);
  };

  const handleSpotSelected = (index: ChampionIndex, teamIndex: TeamIndex) => {
    // If no spot is currently selected, select the new spot
    if (!selectedSpot) {
      setSelectedSpot({ championIndex: index, teamIndex });
      return;
    }

    // If the same spot is clicked again, deselect it
    if (
      selectedSpot.teamIndex === teamIndex &&
      selectedSpot.championIndex === index
    ) {
      setSelectedSpot(null);
      return;
    }

    setSelectedSpot({ championIndex: index, teamIndex });
  };

  const handleAddChampion = (champion: Champion) => {
    if (!selectedSpot) return;
    const newTeam = [...team];
    newTeam[selectedSpot.championIndex] = champion;
    setTeam(newTeam);
    setSelectedSpot(null);
  };

  return (
    <div className="p-4 bg-background text-foreground">
      <h1 className="text-3xl font-bold mb-4">Draft King Desktop</h1>
      <div className="flex gap-4">
        <div className="w-1/4">
          <TeamPanel
            team={team}
            is_first_team={true}
            selectedSpot={selectedSpot}
            onDeleteChampion={handleDeleteChampion}
            onSpotSelected={handleSpotSelected}
            ImageComponent={PlainImage}
          />
        </div>
        <div className="w-1/2">
          <ChampionGrid
            champions={champions}
            addChampion={handleAddChampion}
            favorites={favorites}
            setFavorites={setFavorites}
            ImageComponent={PlainImage}
          />
        </div>
      </div>
      <Button variant="outline" className="mt-4">
        Click me please UwU
      </Button>
    </div>
  );
}

export default App;
