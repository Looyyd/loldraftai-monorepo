import type {
  Team,
  Champion,
  ChampionIndex,
  TeamIndex,
  SelectedSpot,
  Elo,
} from "./types";
import {
  getChampionPlayRates,
  roleToIndexMap,
  type PlayRates,
} from "./champions";

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

export const eloToNumerical = (elo: Elo): number => {
  const eloMap: Record<Elo, number> = {
    silver: 6,
    gold: 5,
    platinum: 4,
    emerald: 3,
    diamond: 1,
    "master +": 0,
    pro: -1,
  };
  return eloMap[elo];
};

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

  // Mark both champions as manually placed if they exist
  if (championFromSelectedSpot) {
    championFromSelectedSpot.isManuallyPlaced = true;
  }
  if (targetChampion) {
    targetChampion.isManuallyPlaced = true;
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
  // Create a copy of the champion with isManuallyPlaced set based on whether we have a selectedSpot
  const championToAdd = {
    ...champion,
    isManuallyPlaced: selectedSpot !== null,
  };

  if (selectedSpot !== null) {
    const team = selectedSpot.teamIndex === 1 ? teamOne : teamTwo;
    let updatedRemainingChampions = handleDeleteChampion(
      selectedSpot.championIndex,
      team
    );

    if (selectedSpot.teamIndex === 1) {
      const newTeam = { ...teamOne };
      newTeam[selectedSpot.championIndex] = championToAdd;
      setTeamOne(newTeam);
    } else {
      const newTeam = { ...teamTwo };
      newTeam[selectedSpot.championIndex] = championToAdd;
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
  const targetTeam = teamToAddToIndex === 0 ? teamOne : teamTwo;

  // Get all champions including the new one
  const allChampions: Champion[] = Object.values(targetTeam).filter(
    (c): c is Champion => c !== undefined
  );
  allChampions.push(championToAdd); // Add our copy with isManuallyPlaced set

  // Get play rates for all champions
  const playRatesMap: Map<number, PlayRates> = new Map();
  for (const champ of allChampions) {
    const rates = getChampionPlayRates(champ.id, currentPatch);
    if (rates) {
      playRatesMap.set(champ.id, rates);
    }
  }

  // Separate manually placed champions from those we can reassign
  const reassignableChampions: Champion[] = allChampions.filter(
    (c) => !c.isManuallyPlaced
  );

  // Generate all possible position assignments
  type Assignment = { [roleIndex: number]: Champion };
  let bestAssignment: Assignment | null = null;
  let bestProbability: number = -1;

  // Function to generate all permutations
  function generateAssignments(
    champions: Champion[],
    currentAssignment: Assignment,
    remainingPositions: number[]
  ): void {
    if (champions.length === 0) {
      // Calculate probability of this assignment
      let probability = 1.0;
      for (const [roleIndex, champ] of Object.entries(currentAssignment)) {
        const index = parseInt(roleIndex);
        const role = Object.keys(roleToIndexMap).find(
          (key) => roleToIndexMap[key] === index
        ) as keyof PlayRates;

        const rates = playRatesMap.get(champ.id);
        const roleRate = rates ? rates[role] : 0.001; // Default to low probability if no data
        probability *= roleRate;
      }

      if (probability > bestProbability) {
        bestProbability = probability;
        bestAssignment = { ...currentAssignment };
      }
      return;
    }

    const currentChamp = champions[0] as Champion;
    const restChampions = champions.slice(1);

    // Try each remaining position for the current champion
    for (let i = 0; i < remainingPositions.length; i++) {
      const roleIndex = remainingPositions[i] as number;
      const updatedAssignment = {
        ...currentAssignment,
        [roleIndex]: currentChamp,
      };
      const updatedPositions = [
        ...remainingPositions.slice(0, i),
        ...remainingPositions.slice(i + 1),
      ];

      generateAssignments(restChampions, updatedAssignment, updatedPositions);
    }
  }

  // Start with an assignment that includes manually placed champions
  const initialAssignment: Assignment = {};
  const occupiedPositions: number[] = [];

  // Add manually placed champions to initial assignment
  for (const [index, champ] of Object.entries(targetTeam)) {
    if (champ && champ.isManuallyPlaced) {
      const roleIndex = parseInt(index);
      initialAssignment[roleIndex] = champ;
      occupiedPositions.push(roleIndex);
    }
  }

  // Determine available positions for reassignable champions
  const allPositions = [0, 1, 2, 3, 4];
  const availablePositions = allPositions.filter(
    (pos) => !occupiedPositions.includes(pos)
  );

  // Generate all possible assignments for reassignable champions
  generateAssignments(
    reassignableChampions,
    initialAssignment,
    availablePositions
  );

  // Apply the best assignment
  if (bestAssignment) {
    const newTeam = { ...emptyTeam };
    for (const [roleIndex, champ] of Object.entries(bestAssignment)) {
      newTeam[parseInt(roleIndex) as keyof Team] = champ as Champion;
    }

    if (teamToAddToIndex === 0) {
      setTeamOne(newTeam);
    } else {
      setTeamTwo(newTeam);
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

  // Reset isManuallyPlaced flag when removing champion
  const championToAdd = { ...champion, isManuallyPlaced: false };

  // Check if the champion is already in the remaining champions list
  const isChampionAlreadyRemaining = remainingChampions.some(
    (remainingChampion) => remainingChampion.id === champion.id
  );

  let champions;
  if (!isChampionAlreadyRemaining) {
    champions = [...remainingChampions, championToAdd].sort((a, b) =>
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
  } else if (team === teamTwo) {
    setTeamTwo(newTeam);
  } else {
    console.error("Team not found in handleDeleteChampion");
  }
  return champions;
}
