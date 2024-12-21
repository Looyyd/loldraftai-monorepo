export interface Champion {
  id: number;
  name: string;
  searchName: string;
  icon: string;
}

export type ChampionIndex = 0 | 1 | 2 | 3 | 4;
export type TeamIndex = 1 | 2;

export type Team = {
  [K in ChampionIndex]: Champion | undefined;
};

export const championIndexToFavoritesPosition = (index: ChampionIndex) => {
  switch (index) {
    case 0:
      return "top";
    case 1:
      return "jungle";
    case 2:
      return "mid";
    case 3:
      return "bot";
    case 4:
      return "support";
  }
};

export type FavoriteChampions = {
  top: number[];
  jungle: number[];
  mid: number[];
  bot: number[];
  support: number[];
};

export interface SelectedSpot {
  teamIndex: TeamIndex;
  championIndex: ChampionIndex;
}

export const elos = [
  "emerald",
  "low diamond",
  "high diamond",
  "master +",
] as const;
export type Elo = (typeof elos)[number];
export const eloToNumerical = (elo: Elo) => elos.indexOf(elo);
