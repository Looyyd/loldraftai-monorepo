import React, { useState, useEffect, useCallback, useRef } from "react";
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
import type {
  ImageComponent,
  Champion,
  FavoriteChampions,
} from "@draftking/ui/lib/types";
import {
  getChampionPlayRates,
  sortedPatches,
  type PlayRates,
} from "@draftking/ui/lib/champions";

interface ChampionGridProps {
  champions: Champion[];
  addChampion: (champion: Champion) => void;
  favorites: FavoriteChampions;
  setFavorites: (favorites: FavoriteChampions) => void;
  ImageComponent: ImageComponent;
}

interface SearchBarProps {
  searchTerm: string;
  setSearchTerm: (searchTerm: string) => void;
  handleKeyDown: (e: React.KeyboardEvent) => void;
  filteredChampions: Champion[];
  inputRef?: React.RefObject<HTMLInputElement>;
}

const SearchBar = ({
  searchTerm,
  setSearchTerm,
  handleKeyDown,
  filteredChampions,
  inputRef,
}: SearchBarProps) => {
  return (
    <div className="w-full relative z-10">
      <TooltipProvider>
        <Tooltip open={filteredChampions.length === 1}>
          <TooltipTrigger asChild>
            <Input
              className="rounded-t ring-2 ring-neutral-950 ring-offset-2 ring-offset-white dark:ring-neutral-300 dark:ring-offset-neutral-950 focus-visible:ring-2 focus-visible:ring-neutral-950 dark:focus-visible:ring-neutral-300"
              type="text"
              placeholder="Search..."
              onChange={(e) => setSearchTerm(e.target.value)}
              value={searchTerm}
              onKeyDown={(e) => handleKeyDown(e)}
              ref={inputRef}
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

const roles = [
  { key: "TOP", label: "Top" },
  { key: "JUNGLE", label: "Jungle" },
  { key: "MIDDLE", label: "Mid" },
  { key: "BOTTOM", label: "Bot" },
  { key: "UTILITY", label: "Support" },
];

export const ChampionGrid: React.FC<ChampionGridProps> = ({
  champions,
  addChampion,
  favorites,
  setFavorites,
  ImageComponent,
}) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedRole, setSelectedRole] = useState<string | null>(null);
  const [showOnlyFavorites, setShowOnlyFavorites] = useState(false);
  const [filteredChampions, setFilteredChampions] = useState(champions);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Initial focus
    inputRef.current?.focus({ preventScroll: true });

    const handleGlobalClick = () => {
      inputRef.current?.focus({ preventScroll: true });
    };

    document.addEventListener("click", handleGlobalClick);
    return () => document.removeEventListener("click", handleGlobalClick);
  }, []);

  // Get the latest patch for play rates
  const latestPatch = sortedPatches[0] as string;

  const isChampionPlayedInRole = useCallback(
    (championId: number, role: string): boolean => {
      // Check if champion is favorited for this role
      const roleMap: Record<string, keyof FavoriteChampions> = {
        TOP: "top",
        JUNGLE: "jungle",
        MIDDLE: "mid",
        BOTTOM: "bot",
        UTILITY: "support",
      };

      const favoriteRole = roleMap[role];
      if (favoriteRole && favorites[favoriteRole].includes(championId)) {
        return true;
      }

      // Check play rates if not favorited
      const playRates = getChampionPlayRates(championId, latestPatch);
      if (!playRates) return false;
      return playRates[role as keyof PlayRates] >= 0.5;
    },
    [latestPatch, favorites]
  );

  const isChampionFavorite = useCallback(
    (championId: number): boolean => {
      return Object.values(favorites).some((roleList) =>
        roleList.includes(championId)
      );
    },
    [favorites]
  );

  const debouncedFilter = useCallback(
    (term: string, role: string | null, favoritesOnly: boolean) => {
      let results = champions.filter((champion) =>
        champion.searchName.includes(term.toLowerCase())
      );

      if (favoritesOnly) {
        results = results.filter((champion) => isChampionFavorite(champion.id));
      }

      if (role) {
        results = results.filter((champion) =>
          isChampionPlayedInRole(champion.id, role)
        );
      }

      setFilteredChampions(results);
    },
    [champions, isChampionPlayedInRole, isChampionFavorite]
  );

  useEffect(() => {
    debouncedFilter(searchTerm, selectedRole, showOnlyFavorites);
  }, [searchTerm, selectedRole, showOnlyFavorites, debouncedFilter]);

  const handleAddToFavorites = (
    champion: Champion,
    position: keyof FavoriteChampions
  ) => {
    const updatedFavorites = {
      ...favorites,
      [position]: [...favorites[position], champion.id],
    };
    setFavorites(updatedFavorites);
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

  const handleRoleClick = (role: string) => {
    setSelectedRole(selectedRole === role ? null : role);
  };

  const handleFavoritesClick = () => {
    setShowOnlyFavorites(!showOnlyFavorites);
  };

  return (
    <div className="h-[560px] rounded border border-gray-200 bg-zinc-800">
      <SearchBar
        searchTerm={searchTerm}
        setSearchTerm={setSearchTerm}
        handleKeyDown={handleKeyDown}
        filteredChampions={filteredChampions}
        inputRef={inputRef}
      />

      {/* Filter buttons */}
      <div className="flex items-center gap-3 p-2 border-b border-border">
        <span className="text-sm text-muted-foreground">Filters:</span>
        <div className="flex gap-2">
          <button
            onClick={handleFavoritesClick}
            className={`
              p-1.5 rounded flex items-center justify-center
              transition-colors duration-200 min-w-[32px]
              ${
                showOnlyFavorites
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary hover:bg-secondary/80 text-secondary-foreground"
              }
            `}
            title="Show Favorites"
          >
            <StarIcon className="h-5 w-5 text-yellow-500" />
          </button>
          {roles.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => handleRoleClick(key)}
              className={`
                p-1.5 rounded flex items-center justify-center
                transition-colors duration-200 min-w-[32px]
                ${
                  selectedRole === key
                    ? "bg-primary text-primary-foreground"
                    : "bg-secondary hover:bg-secondary/80 text-secondary-foreground"
                }
              `}
              title={key}
            >
              <ImageComponent
                src={`/icons/roles/Position_Challenger-${label}.png`}
                alt={label}
                width={20}
                height={20}
                className="w-5 h-5"
              />
            </button>
          ))}
        </div>
      </div>

      <div className="h-[455px] overflow-y-auto p-1">
        <div className="grid grid-cols-[repeat(auto-fill,80px)] justify-center gap-2">
          {filteredChampions.map((champion) => (
            <ContextMenu key={champion.id}>
              <ContextMenuTrigger>
                <div
                  onClick={() => {
                    addChampion(champion);
                    setSearchTerm("");
                  }}
                  className="relative cursor-pointer hover:scale-110 hover:brightness-110"
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
                  {["top", "jungle", "mid", "bot", "support"].map(
                    (position) => {
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
                                position.charAt(0).toUpperCase() +
                                position.slice(1)
                              } Favorites`
                            : `Add to ${
                                position.charAt(0).toUpperCase() +
                                position.slice(1)
                              } Favorites`}
                        </ContextMenuItem>
                      );
                    }
                  )}
                </ContextMenuContent>
              </ContextMenuPortal>
            </ContextMenu>
          ))}
        </div>
      </div>
    </div>
  );
};
