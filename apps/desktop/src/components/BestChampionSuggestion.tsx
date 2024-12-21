import { BestChampionSuggestion as SharedBestChampionSuggestion } from "@draftking/ui/components/draftking/BestChampionSuggestion";
import type { Champion, Team, SelectedSpot, FavoriteChampions, Elo } from "@draftking/ui/lib/types";
import { WinrateBar } from "./WinrateBar";

interface BestChampionSuggestionProps {
  team1: Team;
  team2: Team;
  selectedSpot: SelectedSpot;
  favorites: FavoriteChampions;
  remainingChampions: Champion[];
  elo: Elo;
  patch: string;
}

export const BestChampionSuggestion = (props: BestChampionSuggestionProps) => {
  return (
    <SharedBestChampionSuggestion
      {...props}
      baseApiUrl="http://localhost:3000"
      WinrateBar={WinrateBar}
    />
  );
}; 