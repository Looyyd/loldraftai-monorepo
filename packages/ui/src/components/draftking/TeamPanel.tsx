import React, { useState } from "react";
import clsx from "clsx";
import { TrashIcon } from "@heroicons/react/16/solid";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from "../ui/context-menu";

// Types that should be moved to a shared types package
export interface Champion {
  name: string;
  icon: string;
}

export type ChampionIndex = 0 | 1 | 2 | 3 | 4;
export type TeamIndex = 1 | 2;
export type Team = (Champion | undefined)[];

export interface SelectedSpot {
  teamIndex: TeamIndex;
  championIndex: ChampionIndex;
}

// Image component interface
interface ImageComponentProps {
  src: string;
  alt: string;
  width: number;
  height: number;
  className?: string;
}
type ImageComponent = React.FC<ImageComponentProps>;

interface TeamPanelProps {
  team: Team;
  is_first_team: boolean;
  onDeleteChampion: (index: ChampionIndex) => void;
  selectedSpot: SelectedSpot | null;
  onSpotSelected: (index: ChampionIndex, team: TeamIndex) => void;
  ImageComponent: ImageComponent;
}

const roles = ["Top", "Jungle", "Mid", "Bot", "Support"];

// Subcomponents remain mostly the same, but now use the passed ImageComponent
const TeamTitle = ({ is_blue_side }: { is_blue_side: boolean }) => (
  <div
    className={clsx("rounded-lg p-4 shadow text-center", {
      "bg-blue-500": is_blue_side,
      "bg-red-500": !is_blue_side,
    })}
  >
    <h5 className="text-lg font-bold text-white">
      {is_blue_side ? "Blue Side" : "Red Side"}
    </h5>
  </div>
);

export const TeamPanel: React.FC<TeamPanelProps> = ({
  team,
  is_first_team,
  onDeleteChampion,
  selectedSpot,
  onSpotSelected,
  ImageComponent,
}) => {
  const pannelTeamIndex = is_first_team ? 1 : 2;
  const [hoveredChampion, setHoveredChampion] = useState<number | null>(null);

  const handleSpotClick = (index: ChampionIndex) => {
    onSpotSelected(index, pannelTeamIndex as TeamIndex);
  };

  const handleDeleteChampion = (
    index: ChampionIndex,
    event: React.MouseEvent
  ) => {
    event.stopPropagation();
    onDeleteChampion(index);
    setHoveredChampion(null);
  };

  return (
    <div
      className={clsx("flex flex-col h-full rounded", {
        "bg-blue-900": is_first_team,
        "bg-red-900": !is_first_team,
      })}
    >
      <div className="flex flex-col flex-1">
        <TeamTitle is_blue_side={is_first_team} />
        <ul className="flex flex-col flex-1 justify-between">
          {roles.map((role, index) => {
            const championIndex = index as ChampionIndex;
            const teamMember = team[championIndex];
            const isSelected =
              selectedSpot?.championIndex === championIndex &&
              selectedSpot?.teamIndex === pannelTeamIndex;

            return (
              <li key={index}>
                <div
                  className={clsx("flex justify-center p-1", {
                    "border-4 border-yellow-300": isSelected,
                    "hover:border-2 hover:border-yellow-300": !isSelected,
                  })}
                  onClick={() => handleSpotClick(championIndex)}
                >
                  <div
                    className={clsx("flex items-center", {
                      "flex-row": is_first_team,
                      "flex-row-reverse": !is_first_team,
                    })}
                  >
                    {!teamMember ? (
                      <ImageComponent
                        src={`/icons/roles/Position_Challenger-${role}.png`}
                        alt={role}
                        width={80}
                        height={80}
                      />
                    ) : (
                      <ContextMenu>
                        <ContextMenuTrigger>
                          <div
                            onMouseEnter={() => setHoveredChampion(championIndex)}
                            onMouseLeave={() => setHoveredChampion(null)}
                            className="flex cursor-pointer items-center justify-between"
                          >
                            <ImageComponent
                              src={`/icons/champions/${teamMember.icon}`}
                              alt={teamMember.name}
                              className="block"
                              width={80}
                              height={80}
                            />
                          </div>
                        </ContextMenuTrigger>
                        <ContextMenuContent>
                          <ContextMenuItem
                            onClick={(event: React.MouseEvent) =>
                              handleDeleteChampion(championIndex, event)
                            }
                          >
                            <TrashIcon className="mr-2 h-4 w-4" />
                            Remove Champion
                          </ContextMenuItem>
                        </ContextMenuContent>
                      </ContextMenu>
                    )}
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}; 