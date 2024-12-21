import { useState } from "react";
import { Button } from "@draftking/ui/components/ui/button";
import { TeamPanel } from "@draftking/ui/components/draftking/TeamPanel";
import { ChampionGrid } from "@draftking/ui/components/draftking/ChampionGrid";
import { AnalysisParent } from "./components/AnalysisParent";
import { HelpModal } from "@draftking/ui/components/draftking/HelpModal";
import {
  champions,
  getChampionRoles,
  roleToIndexMap,
} from "@draftking/ui/lib/champions";
import { useDraftStore } from "./stores/draftStore";
import type {
  Team,
  SelectedSpot,
  ChampionIndex,
  TeamIndex,
  Champion,
  FavoriteChampions,
  Elo,
} from "@draftking/ui/lib/types";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@draftking/ui/components/ui/select";

// Plain image component for Electron
const PlainImage: React.FC<{
  src: string;
  alt: string;
  width: number;
  height: number;
  className?: string;
}> = (props) => <img {...props} />;

const emptyTeam: Team = [undefined, undefined, undefined, undefined, undefined];

const DRAFT_ORDERS = {
  "Draft Order": [0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
  "Blue then Red": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "Red then Blue": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
} as const;

type DraftOrderKey = keyof typeof DRAFT_ORDERS;

function getNextPickingTeam(
  teamOne: Team,
  teamTwo: Team,
  pickOrder: readonly number[]
): "BLUE" | "RED" | null {
  const teamOneLength = Object.values(teamOne).filter(
    (c) => c !== undefined
  ).length;
  const teamTwoLength = Object.values(teamTwo).filter(
    (c) => c !== undefined
  ).length;
  const championsPicked = teamOneLength + teamTwoLength;

  if (championsPicked >= 10) return null;

  if (pickOrder[championsPicked] === 0) {
    return teamOneLength >= 5 ? "RED" : "BLUE";
  } else {
    return teamTwoLength >= 5 ? "BLUE" : "RED";
  }
}

function App() {
  // Draft state
  const [teamOne, setTeamOne] = useState<Team>(emptyTeam);
  const [teamTwo, setTeamTwo] = useState<Team>(emptyTeam);
  const [selectedSpot, setSelectedSpot] = useState<SelectedSpot | null>(null);
  const [showHelpModal, setShowHelpModal] = useState(false);
  const [favorites, setFavorites] = useState<FavoriteChampions>({
    top: [],
    jungle: [],
    mid: [],
    bot: [],
    support: [],
  });
  const [elo, setElo] = useState<Elo>("emerald");
  const [selectedDraftOrder, setSelectedDraftOrder] =
    useState<DraftOrderKey>("Draft Order");
  const [remainingChampions, setRemainingChampions] =
    useState<Champion[]>(champions);

  // Store
  const { currentPatch, patches, setCurrentPatch, setPatchList } =
    useDraftStore();

  // Handlers
  const handleDeleteChampion = (index: ChampionIndex, team: Team) => {
    const champion = team[index];
    if (champion && !remainingChampions.some((c) => c.id === champion.id)) {
      setRemainingChampions(
        [...remainingChampions, champion].sort((a, b) =>
          a.name.localeCompare(b.name)
        )
      );
    }

    const newTeam = [...team];
    newTeam[index] = undefined;
    if (team === teamOne) {
      setTeamOne(newTeam);
    } else {
      setTeamTwo(newTeam);
    }
  };

  const handleSpotSelected = (index: ChampionIndex, teamIndex: TeamIndex) => {
    if (!selectedSpot) {
      setSelectedSpot({ championIndex: index, teamIndex });
      return;
    }

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
    if (selectedSpot) {
      // If a spot is selected, add champion there
      const newTeam =
        selectedSpot.teamIndex === 1 ? [...teamOne] : [...teamTwo];
      newTeam[selectedSpot.championIndex] = champion;

      if (selectedSpot.teamIndex === 1) {
        setTeamOne(newTeam);
      } else {
        setTeamTwo(newTeam);
      }
      setSelectedSpot(null);

      // Update remaining champions
      setRemainingChampions(
        remainingChampions.filter((c) => c.id !== champion.id)
      );
      return;
    }

    // Otherwise, follow draft order
    const nextTeam = getNextPickingTeam(
      teamOne,
      teamTwo,
      DRAFT_ORDERS[selectedDraftOrder]
    );
    if (!nextTeam) return;

    // Get the team to add to
    const team = nextTeam === "BLUE" ? teamOne : teamTwo;
    const setTeam = nextTeam === "BLUE" ? setTeamOne : setTeamTwo;

    // Get champion's preferred roles
    const potentialRoles = getChampionRoles(champion.id, currentPatch);
    const potentialRolesIndexes = potentialRoles.map(
      (role) => roleToIndexMap[role]
    );

    // Add rest of indexes at the end of potentialRolesIndexes
    for (let i = 0; i < 5; i++) {
      if (!potentialRolesIndexes.includes(i)) {
        potentialRolesIndexes.push(i);
      }
    }

    const newTeam = [...team];

    // Try to place champion in their preferred role first
    for (const roleIndex of potentialRolesIndexes) {
      if (!newTeam[roleIndex]) {
        newTeam[roleIndex] = champion;
        setTeam(newTeam);

        // Update remaining champions
        setRemainingChampions(
          remainingChampions.filter((c) => c.id !== champion.id)
        );
        break;
      }
    }
  };

  const getStatusMessage = () => {
    if (selectedSpot) {
      const team = selectedSpot.teamIndex === 1 ? "BLUE" : "RED";
      const teamClass =
        selectedSpot.teamIndex === 1 ? "text-blue-500" : "text-red-500";
      return (
        <span>
          Next Pick: <span className={teamClass}>{team}</span> SELECTED SPOT
        </span>
      );
    }
    const nextTeam = getNextPickingTeam(
      teamOne,
      teamTwo,
      DRAFT_ORDERS[selectedDraftOrder]
    );
    if (!nextTeam) return "Draft Complete";

    const teamClass = nextTeam === "BLUE" ? "text-blue-500" : "text-red-500";
    return (
      <span>
        Next Pick: <span className={teamClass}>{nextTeam}</span> TEAM
      </span>
    );
  };

  const resetDraft = () => {
    setTeamOne(emptyTeam);
    setTeamTwo(emptyTeam);
    setSelectedSpot(null);
    setRemainingChampions(champions);
  };

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      <div className="flex gap-2 p-4">
        <Button variant="outline" onClick={resetDraft}>
          Reset Draft
        </Button>
        <Select
          value={selectedDraftOrder}
          onValueChange={(value: DraftOrderKey) => setSelectedDraftOrder(value)}
        >
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Select draft order" />
          </SelectTrigger>
          <SelectContent>
            {Object.keys(DRAFT_ORDERS).map((order) => (
              <SelectItem key={order} value={order}>
                {order}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button variant="outline" onClick={() => setShowHelpModal(true)}>
          Help
        </Button>
      </div>

      <HelpModal
        isOpen={showHelpModal}
        closeHandler={() => setShowHelpModal(false)}
      />

      <div className="text-center text-lg font-semibold mb-4">
        {getStatusMessage()}
      </div>

      <div className="flex flex-1 gap-4 p-4">
        <div className="w-1/4">
          <TeamPanel
            team={teamOne}
            is_first_team={true}
            selectedSpot={selectedSpot}
            onDeleteChampion={(index) => handleDeleteChampion(index, teamOne)}
            onSpotSelected={handleSpotSelected}
            ImageComponent={PlainImage}
          />
        </div>

        <div className="w-1/2">
          <ChampionGrid
            champions={remainingChampions}
            addChampion={handleAddChampion}
            favorites={favorites}
            setFavorites={setFavorites}
            ImageComponent={PlainImage}
          />
        </div>

        <div className="w-1/4">
          <TeamPanel
            team={teamTwo}
            is_first_team={false}
            selectedSpot={selectedSpot}
            onDeleteChampion={(index) => handleDeleteChampion(index, teamTwo)}
            onSpotSelected={handleSpotSelected}
            ImageComponent={PlainImage}
          />
        </div>
      </div>

      <div className="p-4">
        <AnalysisParent
          team1={teamOne}
          team2={teamTwo}
          selectedSpot={selectedSpot}
          favorites={favorites}
          remainingChampions={remainingChampions}
          analysisTrigger={0}
          elo={elo}
          setElo={setElo}
          currentPatch={currentPatch}
          patches={patches}
          setCurrentPatch={setCurrentPatch}
          setPatchList={setPatchList}
        />
      </div>
    </div>
  );
}

export default App;
