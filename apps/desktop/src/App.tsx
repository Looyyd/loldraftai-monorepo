import { useState, useEffect, useMemo } from "react";
import { Button } from "@draftking/ui/components/ui/button";
import { TeamPanel } from "@draftking/ui/components/draftking/TeamPanel";
import { ChampionGrid } from "@draftking/ui/components/draftking/ChampionGrid";
import { AnalysisParent } from "./components/AnalysisParent";
import { PlainImage } from "./components/PlainImage";
import { HelpModal } from "@draftking/ui/components/draftking/HelpModal";
import { champions, roleToIndexMap } from "@draftking/ui/lib/champions";
import { useDraftStore } from "./stores/draftStore";
import type {
  Team,
  SelectedSpot,
  ChampionIndex,
  TeamIndex,
  Champion,
  FavoriteChampions,
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
import { getChampionRoles } from "@draftking/ui/lib/champions";
import { StatusMessage } from "@draftking/ui/components/draftking/StatusMessage";
import { useToast } from "@draftking/ui/hooks/use-toast";
import { usePersistedState } from "@draftking/ui/hooks/usePersistedState";
import isEqual from "lodash/isEqual";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "@draftking/ui/components/ui/tooltip";

function App() {
  // Draft state
  const [teamOne, setTeamOne] = useState<Team>(emptyTeam);
  const [teamTwo, setTeamTwo] = useState<Team>(emptyTeam);
  const [selectedSpot, setSelectedSpot] = useState<SelectedSpot | null>(null);
  const [showHelpModal, setShowHelpModal] = useState(false);
  const [favorites, setFavorites] = usePersistedState<FavoriteChampions>(
    "favorites",
    {
      top: [],
      jungle: [],
      mid: [],
      bot: [],
      support: [],
    }
  );
  const [selectedDraftOrder, setSelectedDraftOrder] =
    useState<DraftOrderKey>("Draft Order");
  const [remainingChampions, setRemainingChampions] =
    useState<Champion[]>(champions);
  const [isLiveTracking, setIsLiveTracking] = useState(false);
  const [bannedChampions, setBannedChampions] = useState<Champion[]>([]);
  const [resetAnalysisTrigger, setResetAnalysisTrigger] = useState(0);

  // Store
  const { currentPatch, patches, setCurrentPatch, setPatchList } =
    useDraftStore();

  const { toast } = useToast();

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
    setBannedChampions([]);
    setRemainingChampions(champions);
    setResetAnalysisTrigger((prev) => prev + 1);
  };

  useEffect(() => {
    // Listen for update notifications
    window.electronAPI.onUpdateNotification((info) => {
      toast({
        title: info.title,
        description: info.body,
      });
    });
  }, [toast]);

  useEffect(() => {
    if (!isLiveTracking) return;

    let intervalId: NodeJS.Timeout;

    async function updateFromLiveGame() {
      try {
        const champSelect = await window.electronAPI.getChampSelect();
        if (!champSelect) {
          setIsLiveTracking(false);
          toast({
            title: "Live Tracking Stopped",
            description: "No live draft lobby found",
          });
          return;
        }

        // Process bans
        const bannedActions = champSelect.actions
          .flat()
          .filter(
            (action: any) => action.type === "ban" && action.completed === true
          );

        // TODO: i think this is not needed, we can just compare all banned champions
        const newBannedChampions = bannedActions
          .map((action: any) =>
            champions.find((c) => c.id === action.championId)
          )
          .filter(
            (champion: any): champion is Champion => champion !== undefined
          );

        // Accumulate bans while avoiding duplicates
        setBannedChampions((current) => {
          const mergedBans = [...current];
          for (const champion of newBannedChampions) {
            if (!mergedBans.some((ban) => ban.id === champion.id)) {
              mergedBans.push(champion);
            }
          }
          return isEqual(current, mergedBans) ? current : mergedBans;
        });

        // Process picks
        const completedActions = champSelect.actions
          .flat()
          .filter(
            (action: any) => action.type === "pick" && action.completed === true
          );

        const allPlayers = [...champSelect.myTeam, ...champSelect.theirTeam];

        // Build new team states first, then update only if changed
        const newTeamOne = { ...teamOne };
        const newTeamTwo = { ...teamTwo };
        let hasChanges = false;

        // Track newly picked champions to update remaining champions
        const newlyPickedChampions: Champion[] = [];

        for (const player of allPlayers) {
          const completedAction = completedActions.find(
            (action: any) => action.actorCellId === player.cellId
          );

          if (!completedAction?.championId) continue;

          const champion = champions.find(
            (c) => c.id === completedAction.championId
          );
          if (!champion) continue;

          const targetTeam = player.team === 1 ? newTeamOne : newTeamTwo;

          // Skip if champion is already in either team
          const isAlreadyInTeams = [
            ...Object.values(newTeamOne),
            ...Object.values(newTeamTwo),
          ].some((c) => c && c.id === champion.id);
          if (isAlreadyInTeams) continue;

          // Get potential roles based on play rates
          const potentialRoles = getChampionRoles(champion.id, currentPatch);
          const potentialRolesIndexes = [
            ...potentialRoles.map((role) => roleToIndexMap[role]),
            ...Array.from({ length: 5 }, (_, i) => i), // Add remaining roles
          ].filter((value, index, self) => self.indexOf(value) === index); // Remove duplicates

          // Try to place champion in their most played role first
          for (const roleIndex of potentialRolesIndexes) {
            if (!targetTeam[roleIndex as keyof Team]) {
              targetTeam[roleIndex as keyof Team] = champion;
              hasChanges = true;
              newlyPickedChampions.push(champion);
              break;
            }
          }
        }

        // Only update teams if there were actual changes
        if (hasChanges) {
          if (!isEqual(teamOne, newTeamOne)) {
            setTeamOne(newTeamOne);
          }
          if (!isEqual(teamTwo, newTeamTwo)) {
            setTeamTwo(newTeamTwo);
          }

          // Update remaining champions if new picks were made
          if (newlyPickedChampions.length > 0) {
            setRemainingChampions((current) => {
              const newRemaining = current.filter(
                (c) =>
                  !newlyPickedChampions.some((picked) => picked.id === c.id)
              );
              return isEqual(current, newRemaining) ? current : newRemaining;
            });
          }
        }

        // Check if draft is complete
        if (champSelect.timer.phase === "GAME_STARTING") {
          setIsLiveTracking(false);
        }
      } catch (error) {
        console.error("Error updating from live game:", error);
        setIsLiveTracking(false);
      }
    }

    void updateFromLiveGame();
    intervalId = setInterval(updateFromLiveGame, 500);

    return () => {
      clearInterval(intervalId);
      setBannedChampions([]);
    };
  }, [isLiveTracking, teamOne, teamTwo, currentPatch]);

  const toggleLiveTracking = () => {
    if (!isLiveTracking) {
      resetDraft();
    }
    setIsLiveTracking(!isLiveTracking);
  };

  const remainingNonBannedChampions = useMemo(() => {
    const bannedIds = new Set(bannedChampions.map((c) => c.id));
    return remainingChampions.filter((c) => !bannedIds.has(c.id));
  }, [bannedChampions, remainingChampions]);

  return (
    <>
      {/* Mobile message */}
      <div className="md:hidden flex mt-16 w-full items-center justify-center p-4">
        <h2 className="text-center text-xl font-semibold text-primary">
          Sorry, <span className="brand-text">LoLDraftAI</span> is not yet
          available on mobile devices. Please use a larger screen.
        </h2>
      </div>

      {/* Main content - hidden on mobile */}
      <div className="hidden md:block">
        <div className="container mx-auto mt-8 font-sans">
          <h1 className="text-5xl font-extrabold tracking-tight leading-tight text-primary text-center mb-8">
            LoLDraftAI Analysis
          </h1>
          <div className="mx-auto">
            <div className="flex flex-wrap items-stretch justify-start mb-4">
              <div className="flex w-full p-1 sm:w-auto">
                <div className="flex-1">
                  <Button variant="outline" onClick={resetDraft}>
                    Reset Draft
                  </Button>
                </div>
              </div>
              <div className="flex w-full p-1 sm:w-auto">
                <div className="flex-1">
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
                </div>
              </div>
              <div className="flex w-full p-1 sm:w-auto">
                <div className="flex-1">
                  <Button
                    variant="outline"
                    onClick={() => setShowHelpModal(true)}
                  >
                    Help
                  </Button>
                </div>
              </div>
              <div className="flex w-full p-1 sm:w-auto">
                <div className="flex-1">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="outline"
                          onClick={toggleLiveTracking}
                          className="inline-flex items-center gap-2"
                        >
                          <span className="relative flex h-2 w-2">
                            {isLiveTracking && (
                              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75" />
                            )}
                            <span
                              className={`relative inline-flex h-2 w-2 rounded-full ${
                                isLiveTracking ? "bg-red-500" : "bg-gray-200"
                              }`}
                            />
                          </span>
                          {isLiveTracking
                            ? "Stop Live Tracking"
                            : "Start Live Tracking"}
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-[300px] whitespace-normal">
                        Syncs with your live draft lobby. Champions are
                        auto-placed based on their common roles, but may need
                        manual adjustment since actual positions aren't known.
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              </div>
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
                    champions={remainingNonBannedChampions}
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
                setSelectedSpot={setSelectedSpot}
                favorites={favorites}
                remainingChampions={remainingNonBannedChampions}
                analysisTrigger={0}
                resetAnalysisTrigger={resetAnalysisTrigger}
                currentPatch={currentPatch}
                patches={patches}
                setCurrentPatch={setCurrentPatch}
                setPatchList={setPatchList}
              />
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
