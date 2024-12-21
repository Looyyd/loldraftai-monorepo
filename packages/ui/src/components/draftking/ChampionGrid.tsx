import React, { useState, useEffect, useCallback } from "react";
import { StarIcon } from "@heroicons/react/24/solid";
import { Input } from "../ui/input";
import {
  ContextMenu,
  ContextMenuTrigger,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuPortal,
} from "../ui/context-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";

// Types that should be moved to a shared types package
export interface Champion {
  id: number;
  name: string;
  icon: string;
  searchName: string;
}

export type FavoriteChampions = {
  top: number[];
  jungle: number[];
  mid: number[];
  bot: number[];
  support: number[];
};

interface ImageComponentProps {
  src: string;
  alt: string;
  width: number;
  height: number;
  className?: string;
}
export type ImageComponent = React.FC<ImageComponentProps>;

interface ChampionGridProps {
  champions: Champion[];
  addChampion: (champion: Champion) => void;
  favorites: FavoriteChampions;
  setFavorites: (favorites: FavoriteChampions) => void;
  ImageComponent: ImageComponent;
  onFavoritesChange?: (favorites: FavoriteChampions) => void;
}

interface SearchBarProps {
  searchTerm: string;
  setSearchTerm: (searchTerm: string) => void;
  handleKeyDown: (e: React.KeyboardEvent) => void;
  filteredChampions: Champion[];
}

const SearchBar = ({
  searchTerm,
  setSearchTerm,
  handleKeyDown,
  filteredChampions,
}: SearchBarProps) => {
  return (
    <div className="w-full relative z-10">
      <TooltipProvider>
        <Tooltip open={filteredChampions.length === 1}>
          <TooltipTrigger asChild>
            <Input
              className="rounded-t"
              type="text"
              placeholder="Search..."
              onChange={(e) => setSearchTerm(e.target.value)}
              value={searchTerm}
              onKeyDown={(e) => handleKeyDown(e)}
            />
          </TooltipTrigger>
          <TooltipContent side="top" align="start">
            Press enter to select {filteredChampions[0]?.name}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  );
};

export const ChampionGrid: React.FC<ChampionGridProps> = ({
  champions,
  addChampion,
  favorites,
  setFavorites,
  ImageComponent,
  onFavoritesChange,
}) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [filteredChampions, setFilteredChampions] = useState(champions);

  const debouncedFilter = useCallback(
    (term: string) => {
      const results = champions.filter((champion) =>
        champion.searchName.includes(term.toLowerCase())
      );
      setFilteredChampions(results);
    },
    [champions]
  );

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      debouncedFilter(searchTerm);
    }, 100);

    return () => clearTimeout(timeoutId);
  }, [searchTerm, debouncedFilter]);

  const handleAddToFavorites = (
    champion: Champion,
    position: keyof FavoriteChampions
  ) => {
    const updatedFavorites = {
      ...favorites,
      [position]: [...favorites[position], champion.id],
    };
    setFavorites(updatedFavorites);
    onFavoritesChange?.(updatedFavorites);
  };

  const handleRemoveFromFavorites = (
    champion: Champion,
    position: keyof FavoriteChampions
  ) => {
    const updatedFavorites = {
      ...favorites,
      [position]: favorites[position].filter((id) => id !== champion.id),
    };
    setFavorites(updatedFavorites);
    onFavoritesChange?.(updatedFavorites);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (
      e.key === "Enter" &&
      filteredChampions.length === 1 &&
      filteredChampions[0]
    ) {
      addChampion(filteredChampions[0]);
      setSearchTerm("");
    }
  };

  return (
    <div className="h-[560px] rounded border border-gray-200 bg-zinc-800">
      <SearchBar
        searchTerm={searchTerm}
        setSearchTerm={setSearchTerm}
        handleKeyDown={handleKeyDown}
        filteredChampions={filteredChampions}
      />
      <div className="h-[505px] overflow-y-auto p-1">
        <div className="grid grid-cols-[repeat(auto-fill,80px)] justify-center gap-2">
          {champions.map((champion) => (
            <ContextMenu key={champion.id}>
              <ContextMenuTrigger
                className={`${
                  filteredChampions.map((c) => c.id).includes(champion.id)
                    ? "relative"
                    : "absolute invisible pointer-events-none"
                }`}
              >
                <div
                  onClick={() => {
                    addChampion(champion);
                    setSearchTerm("");
                  }}
                  className="relative cursor-pointer"
                >
                  {Object.entries(favorites).some(([, list]) =>
                    list.includes(champion.id)
                  ) && (
                    <div className="absolute right-0 top-0">
                      <StarIcon
                        className="h-6 w-6 text-yellow-500"
                        stroke="black"
                        strokeWidth={2}
                      />
                    </div>
                  )}
                  <ImageComponent
                    src={`/icons/champions/${champion.icon}`}
                    alt={champion.name}
                    width={80}
                    height={80}
                  />
                </div>
              </ContextMenuTrigger>
              <ContextMenuPortal>
                <ContextMenuContent>
                  {["top", "jungle", "mid", "bot", "support"].map((position) => {
                    const isFavorite = favorites[
                      position as keyof FavoriteChampions
                    ].includes(champion.id);
                    return (
                      <ContextMenuItem
                        key={position}
                        onClick={() =>
                          isFavorite
                            ? handleRemoveFromFavorites(
                                champion,
                                position as keyof FavoriteChampions
                              )
                            : handleAddToFavorites(
                                champion,
                                position as keyof FavoriteChampions
                              )
                        }
                      >
                        <StarIcon
                          className={`mr-2 h-5 w-5 ${
                            isFavorite ? "text-yellow-500" : "text-white"
                          }`}
                          stroke="black"
                          strokeWidth={2}
                        />
                        {isFavorite
                          ? `Remove from ${
                              position.charAt(0).toUpperCase() + position.slice(1)
                            } Favorites`
                          : `Add to ${
                              position.charAt(0).toUpperCase() + position.slice(1)
                            } Favorites`}
                      </ContextMenuItem>
                    );
                  })}
                </ContextMenuContent>
              </ContextMenuPortal>
            </ContextMenu>
          ))}
        </div>
      </div>
    </div>
  );
}; 