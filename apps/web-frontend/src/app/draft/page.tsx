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
import { champions, roleToIndexMap, getChampionRoles } from "@/app/champions";
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

const emptyTeam: Team = {
  0: undefined,
  1: undefined,
  2: undefined,
  3: undefined,
  4: undefined,
};

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

export default function Draft() {
  const [remainingChampions, setRemainingChampions] =
    useState<Champion[]>(champions);
  const [teamOne, setTeamOne] = useState<Team>(emptyTeam);
  const [teamTwo, setTeamTwo] = useState<Team>(emptyTeam);
  const [analysisTrigger] = useState(0);
  const teams = [teamOne, teamTwo];
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
    // If no spot is currently selected, select the new spot
    if (!selectedSpot) {
      setSelectedSpot({ teamIndex: team, championIndex: index });
      return;
    }

    // If the same spot is clicked again, deselect it
    if (
      selectedSpot.teamIndex === team &&
      selectedSpot.championIndex === index
    ) {
      setSelectedSpot(null);
      return;
    }

    const teamOneCopy = { ...teamOne };
    const teamTwoCopy = { ...teamTwo };
    const championFromSelectedSpot =
      selectedSpot.teamIndex === 1
        ? teamOneCopy[selectedSpot.championIndex]
        : teamTwoCopy[selectedSpot.championIndex];
    const targetChampion = team === 1 ? teamOneCopy[index] : teamTwoCopy[index];

    //if both spots are empty, select the new spot
    if (
      championFromSelectedSpot === undefined &&
      targetChampion === undefined
    ) {
      setSelectedSpot({ teamIndex: team, championIndex: index });
      return;
    }

    // Swap the champions between the two spots
    if (selectedSpot.teamIndex === 1) {
      if (team === 1) {
        // Swap within the same team
        teamOneCopy[selectedSpot.championIndex] = targetChampion;
        teamOneCopy[index] = championFromSelectedSpot;
        setTeamOne(teamOneCopy);
      } else {
        // Swap between different teams
        teamOneCopy[selectedSpot.championIndex] = targetChampion;
        teamTwoCopy[index] = championFromSelectedSpot;
        setTeamOne(teamOneCopy);
        setTeamTwo(teamTwoCopy);
      }
    } else {
      if (team === 2) {
        // Swap within the same team
        teamTwoCopy[selectedSpot.championIndex] = targetChampion;
        teamTwoCopy[index] = championFromSelectedSpot;
        setTeamTwo(teamTwoCopy);
      } else {
        // Swap between different teams
        teamTwoCopy[selectedSpot.championIndex] = targetChampion;
        teamOneCopy[index] = championFromSelectedSpot;
        setTeamOne(teamOneCopy);
        setTeamTwo(teamTwoCopy);
      }
    }

    // Reset the selected spot
    setSelectedSpot(null);
  };

  const addChampion = (champion: Champion) => {
    if (selectedSpot !== null) {
      const team = teams[selectedSpot.teamIndex - 1];

      let updatedRemainingChampions = handleDeleteChampion(
        selectedSpot.championIndex,
        team
      );

      if (selectedSpot.teamIndex === 1) {
        const newTeam = { ...teamOne };
        newTeam[selectedSpot.championIndex] = champion;
        setTeamOne(newTeam);
      } else {
        const newTeam = { ...teamTwo };
        newTeam[selectedSpot.championIndex] = champion;
        setTeamTwo(newTeam);
      }
      updatedRemainingChampions = updatedRemainingChampions.filter(
        (c) => c.id !== champion.id
      );
      setRemainingChampions(updatedRemainingChampions);

      setSelectedSpot(null);
      return;
    }

    const pickOrder = DRAFT_ORDERS[selectedDraftOrder];
    const nextTeam = getNextPickingTeam(teamOne, teamTwo, pickOrder);
    if (!nextTeam) return;

    const champions = remainingChampions.filter((c) => c.id !== champion.id);
    setRemainingChampions(champions);

    const teamToAddToIndex = nextTeam === "BLUE" ? 0 : 1;
    const potentialRoles = getChampionRoles(champion.id, currentPatch);

    const potentialRolesIndexes = potentialRoles.map(
      (role) => roleToIndexMap[role]
    );
    // add rest of indexes at the end of potentialRolesIndexes
    for (let i = 0; i < 5; i++) {
      if (!potentialRolesIndexes.includes(i)) {
        potentialRolesIndexes.push(i);
      }
    }
    // add to first available spot
    for (let i = 0; i < 5; i++) {
      const roleIndex = potentialRolesIndexes[i];
      if (teamToAddToIndex === 0 && !teamOne[roleIndex as keyof Team]) {
        setTeamOne({
          ...teams[teamToAddToIndex],
          [roleIndex as keyof Team]: champion,
        });
        return;
      } else if (teamToAddToIndex === 1 && !teamTwo[roleIndex as keyof Team]) {
        setTeamTwo({
          ...teams[teamToAddToIndex],
          [roleIndex as keyof Team]: champion,
        });
        return;
      }
    }
  };

  const handleDeleteChampion = (index: ChampionIndex, team: Team) => {
    const champion = team[index];
    if (champion === undefined) {
      return remainingChampions;
    }
    // Check if the champion is already in the remaining champions list
    const isChampionAlreadyRemaining = remainingChampions.some(
      (remainingChampion) => remainingChampion.id === champion.id
    );
    let champions;
    if (!isChampionAlreadyRemaining) {
      champions = [...remainingChampions, champion].sort((a, b) =>
        a.name.localeCompare(b.name)
      );
      setRemainingChampions(champions);
    } else {
      champions = [...remainingChampions];
    }

    if (team === teamOne) {
      const newTeam = { ...teamOne };
      delete newTeam[index];
      setTeamOne(newTeam);
    } else {
      const newTeam = { ...teamTwo };
      delete newTeam[index];
      setTeamTwo(newTeam);
    }
    return champions;
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

  return (
    <main className="flex  w-full flex-col items-center">
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

          <Button variant="outline" onClick={openHelpModal}>
            Help
          </Button>
        </div>
        <HelpModal isOpen={showHelpModal} closeHandler={closeHelpModal} />

        <div className="text-center text-lg font-semibold mb-4">
          {getStatusMessage()}
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
