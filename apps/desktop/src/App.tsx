import { useState } from "react";
import { Button } from "@draftking/ui/components/ui/button";
import { TeamPanel } from "@draftking/ui/components/draftking/TeamPanel";
import { ChampionGrid } from "@draftking/ui/components/draftking/ChampionGrid";
import { AnalysisParent } from "./components/AnalysisParent";
import { HelpModal } from "@draftking/ui/components/draftking/HelpModal";
import { champions } from "@draftking/ui/lib/champions";
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
import {
  emptyTeam,
  DRAFT_ORDERS,
  handleSpotSelection,
  addChampion as addChampionLogic,
  handleDeleteChampion as handleDeleteChampionLogic,
  type DraftOrderKey,
} from "@draftking/ui/lib/draftLogic";
import { StatusMessage } from "@draftking/ui/components/draftking/StatusMessage";

// Plain image component for Electron
const PlainImage: React.FC<{
  src: string;
  alt: string;
  width: number;
  height: number;
  className?: string;
}> = (props) => <img {...props} />;

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
    return handleDeleteChampionLogic(
      index,
      team,
      teamOne,
      teamTwo,
      remainingChampions,
      setTeamOne,
      setTeamTwo,
      setRemainingChampions
    );
  };

  const handleSpotSelected = (index: ChampionIndex, team: TeamIndex) => {
    handleSpotSelection(
      index,
      team,
      selectedSpot,
      teamOne,
      teamTwo,
      setTeamOne,
      setTeamTwo,
      setSelectedSpot
    );
  };

  const handleAddChampion = (champion: Champion) => {
    addChampionLogic(
      champion,
      selectedSpot,
      teamOne,
      teamTwo,
      remainingChampions,
      currentPatch,
      selectedDraftOrder,
      setTeamOne,
      setTeamTwo,
      setRemainingChampions,
      setSelectedSpot,
      handleDeleteChampion
    );
  };

  const resetDraft = () => {
    setTeamOne(emptyTeam);
    setTeamTwo(emptyTeam);
    setSelectedSpot(null);
    setRemainingChampions(champions);
  };

  return (
    <div className="container mx-auto mt-12">
      <div className="mx-auto">
        <div className="flex gap-2 mb-4">
          <Button variant="outline" onClick={resetDraft}>
            Reset Draft
          </Button>
          <Select
            value={selectedDraftOrder}
            onValueChange={(value: DraftOrderKey) =>
              setSelectedDraftOrder(value)
            }
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
          <StatusMessage
            selectedSpot={selectedSpot}
            teamOne={teamOne}
            teamTwo={teamTwo}
            selectedDraftOrder={selectedDraftOrder}
          />
        </div>

        <div className="flex flex-wrap items-stretch justify-evenly">
          <div className="flex w-full justify-between">
            <div className="flex w-auto max-w-xs p-1">
              <TeamPanel
                team={teamOne}
                is_first_team={true}
                selectedSpot={selectedSpot}
                onDeleteChampion={(index) =>
                  handleDeleteChampion(index, teamOne)
                }
                onSpotSelected={handleSpotSelected}
                ImageComponent={PlainImage}
              />
            </div>

            <div className="grow p-1">
              <ChampionGrid
                champions={remainingChampions}
                addChampion={handleAddChampion}
                favorites={favorites}
                setFavorites={setFavorites}
                ImageComponent={PlainImage}
              />
            </div>

            <div className="flex w-auto max-w-xs p-1">
              <TeamPanel
                team={teamTwo}
                is_first_team={false}
                selectedSpot={selectedSpot}
                onDeleteChampion={(index) =>
                  handleDeleteChampion(index, teamTwo)
                }
                onSpotSelected={handleSpotSelected}
                ImageComponent={PlainImage}
              />
            </div>
          </div>
        </div>

        <div className="mt-4">
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
    </div>
  );
}

export default App;
