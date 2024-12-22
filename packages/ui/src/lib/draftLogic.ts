import type {
  Team,
  Champion,
  ChampionIndex,
  TeamIndex,
  SelectedSpot,
} from "./types";
import { getChampionRoles, roleToIndexMap } from "./champions";
import { ReactNode } from "react";

export const emptyTeam: Team = {
  0: undefined,
  1: undefined,
  2: undefined,
  3: undefined,
  4: undefined,
};

export const DRAFT_ORDERS = {
  "Draft Order": [0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
  "Blue then Red": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "Red then Blue": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
} as const;

export type DraftOrderKey = keyof typeof DRAFT_ORDERS;

export function getNextPickingTeam(
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

export function handleSpotSelection(
  index: ChampionIndex,
  team: TeamIndex,
  selectedSpot: SelectedSpot | null,
  teamOne: Team,
  teamTwo: Team,
  setTeamOne: (team: Team) => void,
  setTeamTwo: (team: Team) => void,
  setSelectedSpot: (spot: SelectedSpot | null) => void
) {
  if (!selectedSpot) {
    setSelectedSpot({ teamIndex: team, championIndex: index });
    return;
  }

  if (selectedSpot.teamIndex === team && selectedSpot.championIndex === index) {
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

  if (championFromSelectedSpot === undefined && targetChampion === undefined) {
    setSelectedSpot({ teamIndex: team, championIndex: index });
    return;
  }

  if (selectedSpot.teamIndex === 1) {
    if (team === 1) {
      teamOneCopy[selectedSpot.championIndex] = targetChampion;
      teamOneCopy[index] = championFromSelectedSpot;
      setTeamOne(teamOneCopy);
    } else {
      teamOneCopy[selectedSpot.championIndex] = targetChampion;
      teamTwoCopy[index] = championFromSelectedSpot;
      setTeamOne(teamOneCopy);
      setTeamTwo(teamTwoCopy);
    }
  } else {
    if (team === 2) {
      teamTwoCopy[selectedSpot.championIndex] = targetChampion;
      teamTwoCopy[index] = championFromSelectedSpot;
      setTeamTwo(teamTwoCopy);
    } else {
      teamTwoCopy[selectedSpot.championIndex] = targetChampion;
      teamOneCopy[index] = championFromSelectedSpot;
      setTeamOne(teamOneCopy);
      setTeamTwo(teamTwoCopy);
    }
  }

  setSelectedSpot(null);
}

export function addChampion(
  champion: Champion,
  selectedSpot: SelectedSpot | null,
  teamOne: Team,
  teamTwo: Team,
  remainingChampions: Champion[],
  currentPatch: string,
  selectedDraftOrder: DraftOrderKey,
  setTeamOne: (team: Team) => void,
  setTeamTwo: (team: Team) => void,
  setRemainingChampions: (champions: Champion[]) => void,
  setSelectedSpot: (spot: SelectedSpot | null) => void,
  handleDeleteChampion: (index: ChampionIndex, team: Team) => Champion[]
) {
  if (selectedSpot !== null) {
    const team = selectedSpot.teamIndex === 1 ? teamOne : teamTwo;
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

  for (let i = 0; i < 5; i++) {
    if (!potentialRolesIndexes.includes(i)) {
      potentialRolesIndexes.push(i);
    }
  }

  for (let i = 0; i < 5; i++) {
    const roleIndex = potentialRolesIndexes[i];
    if (teamToAddToIndex === 0 && !teamOne[roleIndex as keyof Team]) {
      setTeamOne({
        ...teamOne,
        [roleIndex as keyof Team]: champion,
      });
      return;
    } else if (teamToAddToIndex === 1 && !teamTwo[roleIndex as keyof Team]) {
      setTeamTwo({
        ...teamTwo,
        [roleIndex as keyof Team]: champion,
      });
      return;
    }
  }
}

export function handleDeleteChampion(
  index: ChampionIndex,
  team: Team,
  teamOne: Team,
  teamTwo: Team,
  remainingChampions: Champion[],
  setTeamOne: (team: Team) => void,
  setTeamTwo: (team: Team) => void,
  setRemainingChampions: (champions: Champion[]) => void
): Champion[] {
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

  const newTeam = { ...team };
  delete newTeam[index];
  if (team === teamOne) {
    setTeamOne(newTeam);
  } else {
    setTeamTwo(newTeam);
  }
  return champions;
}
