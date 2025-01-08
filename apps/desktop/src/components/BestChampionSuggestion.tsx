import { BestChampionSuggestion as SharedBestChampionSuggestion } from "@draftking/ui/components/draftking/BestChampionSuggestion";
import type {
  Champion,
  Team,
  SelectedSpot,
  FavoriteChampions,
  Elo,
} from "@draftking/ui/lib/types";
import { VERCEL_URL } from "../utils";
import { PlainImage } from "./PlainImage";

interface BestChampionSuggestionProps {
  team1: Team;
  team2: Team;
  selectedSpot: SelectedSpot;
  favorites: FavoriteChampions;
  remainingChampions: Champion[];
  elo: Elo;
  patch: string;
  suggestionMode: "favorites" | "meta" | "all";
}

export const BestChampionSuggestion = (props: BestChampionSuggestionProps) => {
  return (
    <SharedBestChampionSuggestion
      {...props}
      baseApiUrl={VERCEL_URL}
      ImageComponent={PlainImage}
    />
  );
};
