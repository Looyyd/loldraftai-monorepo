"use client";

import React, { useEffect, useState } from "react";

import ChampionGrid from "./ChampionGrid";
import TeamPanel from "./TeamPanel";
import { HelpModal } from "./HelpModal";
import AnalysisParent from "./AnalysisParent";
import type {
  Champion,
  Team,
  TeamIndex,
  ChampionIndex,
  SelectedSpot,
  FavoriteChampions,
} from "@/app/types";
import { champions } from "@/app/champions";
import { Button } from "@draftking/ui/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@draftking/ui/components/ui/select";
import { useDraftStore } from "@/app/stores/draftStore";
import Cookies from "js-cookie";
import {
  emptyTeam,
  DRAFT_ORDERS,
  handleSpotSelection as handleSpotSelectionLogic,
  addChampion as addChampionLogic,
  handleDeleteChampion as handleDeleteChampionLogic,
  type DraftOrderKey,
} from "@draftking/ui/lib/draftLogic";
import { StatusMessage } from "@draftking/ui/components/draftking/StatusMessage";

export default function Draft() {
  const [remainingChampions, setRemainingChampions] =
    useState<Champion[]>(champions);
  const [teamOne, setTeamOne] = useState<Team>(emptyTeam);
  const [teamTwo, setTeamTwo] = useState<Team>(emptyTeam);
  const [analysisTrigger] = useState(0);
  const [selectedSpot, setSelectedSpot] = useState<SelectedSpot | null>(null);
  const [favorites, setFavorites] = useState<FavoriteChampions>({
    top: [],
    jungle: [],
    mid: [],
    bot: [],
    support: [],
  });

  const [showHelpModal, setShowHelpModal] = useState(false);
  const [selectedDraftOrder, setSelectedDraftOrder] =
    useState<DraftOrderKey>("Draft Order");
  const { currentPatch } = useDraftStore();

  const openHelpModal = () => setShowHelpModal(true);
  const closeHelpModal = () => setShowHelpModal(false);

  useEffect(() => {
    // Existing code to initialize favorites
    const savedFavorites = Cookies.get("favorites");
    if (savedFavorites) {
      setFavorites(JSON.parse(savedFavorites) as FavoriteChampions);
    }
  }, [remainingChampions]);

  const resetDraft = () => {
    setRemainingChampions(champions);
    setTeamOne(emptyTeam);
    setTeamTwo(emptyTeam);
    setSelectedSpot(null);
  };

  const handleSpotSelection = (index: ChampionIndex, team: TeamIndex) => {
    handleSpotSelectionLogic(
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

  const addChampion = (champion: Champion) => {
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

  return (
    <main className="flex w-full flex-col items-center">
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
              <Button variant="outline" onClick={openHelpModal}>
                Help
              </Button>
            </div>
          </div>
        </div>
        <HelpModal isOpen={showHelpModal} closeHandler={closeHelpModal} />

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
            {/* Team Panel 1 */}
            <div className="flex w-auto max-w-xs p-1">
              <TeamPanel
                team={teamOne}
                is_first_team={true}
                onDeleteChampion={(index: ChampionIndex) =>
                  handleDeleteChampion(index, teamOne)
                }
                selectedSpot={selectedSpot}
                onSpotSelected={handleSpotSelection}
              />
            </div>

            {/* Champion Grid */}
            <div className="grow p-1">
              <ChampionGrid
                champions={remainingChampions}
                addChampion={addChampion}
                favorites={favorites}
                setFavorites={setFavorites}
              />
            </div>

            {/* Team Panel 2 */}
            <div className="flex w-auto max-w-xs p-1">
              <TeamPanel
                team={teamTwo}
                is_first_team={false}
                onDeleteChampion={(index: ChampionIndex) =>
                  handleDeleteChampion(index, teamTwo)
                }
                selectedSpot={selectedSpot}
                onSpotSelected={handleSpotSelection}
              />
            </div>
          </div>

          {/* Draft Analysis */}
          <div className="flex w-full justify-center">
            <AnalysisParent
              team1={teamOne}
              team2={teamTwo}
              selectedSpot={selectedSpot}
              favorites={favorites}
              remainingChampions={remainingChampions}
              analysisTrigger={analysisTrigger}
            />
          </div>
        </div>
      </div>
    </main>
  );
}
