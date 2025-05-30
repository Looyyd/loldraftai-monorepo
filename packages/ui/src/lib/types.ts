export interface Champion {
  id: number;
  name: string;
  searchName: string;
  icon: string;
  isManuallyPlaced?: boolean;
}

export interface ImageComponentProps {
  src: string;
  alt: string;
  width: number;
  height: number;
  className?: string;
}
export type ImageComponent = React.FC<ImageComponentProps>;

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

export type SelectedSpot = {
  teamIndex: TeamIndex;
  championIndex: ChampionIndex;
};

export const elos = [
  "silver",
  "gold",
  "platinum",
  "emerald",
  "diamond",
  "master +",
] as const;

export const proElos = ["pro"] as const;

export type Elo = (typeof elos)[number] | (typeof proElos)[number];

export type SuggestionMode = "favorites" | "meta" | "all";
export interface DetailedPrediction {
  win_probability: number;
  gold_diff_15min: number[];
  champion_impact: number[];
  time_bucketed_predictions: Record<string, number>;
  raw_time_bucketed_predictions: Record<string, number>;
}
