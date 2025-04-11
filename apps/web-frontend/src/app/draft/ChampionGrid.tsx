"use client";
import React from "react";
import { ChampionGrid as SharedChampionGrid } from "@draftking/ui/components/draftking/ChampionGrid";
import type { Champion, FavoriteChampions } from "@draftking/ui/lib/types";
import { ImageComponent } from "@draftking/ui/lib/types";
import CloudFlareImage from "@/components/CloudFlareImage";

interface ChampionGridProps {
  champions: Champion[];
  addChampion: (champion: Champion) => void;
  favorites: FavoriteChampions;
  setFavorites: (favorites: FavoriteChampions) => void;
}

const ChampionGrid: React.FC<ChampionGridProps> = (props) => {
  return (
    <SharedChampionGrid
      {...props}
      ImageComponent={CloudFlareImage as ImageComponent}
      setFavorites={props.setFavorites}
    />
  );
};

export default ChampionGrid;
