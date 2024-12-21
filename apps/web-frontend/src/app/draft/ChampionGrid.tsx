"use client";
import React from "react";
import Image from "next/image";
import { ChampionGrid as SharedChampionGrid } from "@draftking/ui/components/draftking/ChampionGrid";
import type {
  Champion,
  FavoriteChampions,
  ImageComponent,
} from "@draftking/ui/components/draftking/ChampionGrid";
import Cookies from "js-cookie";

interface ChampionGridProps {
  champions: Champion[];
  addChampion: (champion: Champion) => void;
  favorites: FavoriteChampions;
  setFavorites: (favorites: FavoriteChampions) => void;
}

const ChampionGrid: React.FC<ChampionGridProps> = (props) => {
  const handleFavoritesChange = (favorites: FavoriteChampions) => {
    Cookies.set("favorites", JSON.stringify(favorites), { expires: 365 });
  };

  return (
    <SharedChampionGrid
      {...props}
      ImageComponent={Image as ImageComponent}
      onFavoritesChange={handleFavoritesChange}
    />
  );
};

export default ChampionGrid;
