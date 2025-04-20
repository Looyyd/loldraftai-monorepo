"use client";

import React, { useState, useMemo } from "react";
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
} from "@draftking/ui/lib/types";
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
import {
  emptyTeam,
  DRAFT_ORDERS,
  handleSpotSelection as handleSpotSelectionLogic,
  addChampion as addChampionLogic,
  handleDeleteChampion as handleDeleteChampionLogic,
  type DraftOrderKey,
} from "@draftking/ui/lib/draftLogic";
import { StatusMessage } from "@draftking/ui/components/draftking/StatusMessage";
import { usePersistedState } from "@draftking/ui/hooks/usePersistedState";
import { ChangelogModal } from "./ChangelogModal";
import Link from "next/link";

export default function Draft() {
  const [remainingChampionsBase, setRemainingChampionsBase] =
    useState<Champion[]>(champions);
  const [teamOne, setTeamOne] = useState<Team>(emptyTeam);
  const [teamTwo, setTeamTwo] = useState<Team>(emptyTeam);
  const [analysisTrigger] = useState(0);
  const [resetAnalysisTrigger, setResetAnalysisTrigger] = useState(0);
  const [selectedSpot, setSelectedSpot] = useState<SelectedSpot | null>(null);
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

  // Memoize the champions passed to ChampionGrid
  // This is just to have similar logic between desktop and web version(in desktop, bans are applied here)
  const remainingChampions = useMemo(() => {
    return remainingChampionsBase; // Add filtering logic here if needed in the future
  }, [remainingChampionsBase]);

  const [showHelpModal, setShowHelpModal] = useState(() => {
    // Check if running in browser environment and if we're on desktop
    if (typeof window !== "undefined") {
      const isDesktop = window.innerWidth >= 768; // md breakpoint is 768px
      const hasVisited = localStorage.getItem("hasVisitedBefore");

      if (!hasVisited && isDesktop) {
        localStorage.setItem("hasVisitedBefore", "true");
        return true;
      }
    }
    return false;
  });
  const [selectedDraftOrder, setSelectedDraftOrder] =
    useState<DraftOrderKey>("Draft Order");
  const { currentPatch } = useDraftStore();

  const currentVersion = "1.0.2";
  const [showChangelogModal, setShowChangelogModal] = useState(() => {
    // Check if running in browser environment
    if (typeof window !== "undefined") {
      const lastSeenVersion = localStorage.getItem("lastSeenVersion");

      if (!lastSeenVersion || lastSeenVersion !== currentVersion) {
        localStorage.setItem("lastSeenVersion", currentVersion);
        return true;
      }
    }
    return false;
  });

  const openHelpModal = () => setShowHelpModal(true);
  const closeHelpModal = () => setShowHelpModal(false);
  const closeChangelogModal = () => setShowChangelogModal(false);

  const resetDraft = () => {
    setTeamOne(emptyTeam);
    setTeamTwo(emptyTeam);
    setSelectedSpot(null);
    setRemainingChampionsBase(champions);
    setResetAnalysisTrigger((prev) => prev + 1);
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
      remainingChampionsBase,
      currentPatch,
      selectedDraftOrder,
      setTeamOne,
      setTeamTwo,
      setRemainingChampionsBase,
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
      remainingChampionsBase,
      setTeamOne,
      setTeamTwo,
      setRemainingChampionsBase
    );
  };

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
      <div className="hidden md:flex w-full flex-col items-center mt-4">
        <div className="container mx-auto">
          <h1 className="brand-text text-5xl font-extrabold tracking-tight leading-tight text-primary text-center mb-8">
            LoLDraftAI Analysis
          </h1>
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
                  How to use
                </Button>
              </div>
            </div>
          </div>
          <HelpModal isOpen={showHelpModal} closeHandler={closeHelpModal} />
          <ChangelogModal
            isOpen={showChangelogModal}
            closeHandler={closeChangelogModal}
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
              {/* Team Panel 1 */}
              <div className="flex w-[140px] p-1">
                <TeamPanel
                  team={teamOne}
                  is_first_team={true}
                  onDeleteChampion={(index: ChampionIndex) =>
                    handleDeleteChampion(index, teamOne)
                  }
                  selectedSpot={selectedSpot}
                  onSpotSelected={handleSpotSelection}
                  setTeam={setTeamOne}
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
              <div className="flex w-[140px] p-1">
                <TeamPanel
                  team={teamTwo}
                  is_first_team={false}
                  onDeleteChampion={(index: ChampionIndex) =>
                    handleDeleteChampion(index, teamTwo)
                  }
                  selectedSpot={selectedSpot}
                  onSpotSelected={handleSpotSelection}
                  setTeam={setTeamTwo}
                />
              </div>
            </div>
          </div>
          {/* Add pro tip here */}
          <div className="text-center mt-4 mb-0">
            <p className="text-sm text-blue-600 dark:text-blue-400">
              Pro tip:{" "}
              <Link href="/download" className="underline hover:text-blue-500">
                Download the LoLDraftAI app
              </Link>{" "}
              for automatic champion live tracking
            </p>
          </div>
          {/* Draft Analysis */}
          <div className="mt-0">
            <AnalysisParent
              team1={teamOne}
              team2={teamTwo}
              selectedSpot={selectedSpot}
              setSelectedSpot={setSelectedSpot}
              favorites={favorites}
              remainingChampions={remainingChampions}
              analysisTrigger={analysisTrigger}
              resetAnalysisTrigger={resetAnalysisTrigger}
            />
          </div>
        </div>
      </div>
    </>
  );
}
