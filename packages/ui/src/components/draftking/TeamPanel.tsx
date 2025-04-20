import React from "react";
import clsx from "clsx";
import { Lock, Unlock, LockOpen, Trash2 } from "lucide-react";
import { LockClosedIcon, LockOpenIcon } from "@heroicons/react/24/solid";
import type {
  Team,
  ChampionIndex,
  TeamIndex,
  SelectedSpot,
  ImageComponent,
} from "@draftking/ui/lib/types";

interface TeamPanelProps {
  team: Team;
  is_first_team: boolean;
  onDeleteChampion: (index: ChampionIndex) => void;
  selectedSpot: SelectedSpot | null;
  onSpotSelected: (index: ChampionIndex, team: TeamIndex) => void;
  setTeam: (team: Team) => void;
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
  setTeam,
  ImageComponent,
}) => {
  const pannelTeamIndex = is_first_team ? 1 : 2;

  const handleSpotClick = (index: ChampionIndex) => {
    onSpotSelected(index, pannelTeamIndex as TeamIndex);
  };

  const handleContextMenu = (
    event: React.MouseEvent,
    championIndex: ChampionIndex
  ) => {
    event.preventDefault(); // Prevent default context menu
    onDeleteChampion(championIndex);
  };

  const toggleManualPlacement = (
    event: React.MouseEvent,
    championIndex: ChampionIndex
  ) => {
    event.stopPropagation(); // Prevent triggering spot selection

    // Get the champion at this index
    const champion = team[championIndex];
    if (!champion) return;

    // Create a new team with the updated champion
    const newTeam = { ...team };
    newTeam[championIndex] = {
      ...champion,
      isManuallyPlaced: !champion.isManuallyPlaced,
    };

    // Update the team
    setTeam(newTeam);
  };

  const handleDelete = (
    event: React.MouseEvent,
    championIndex: ChampionIndex
  ) => {
    event.stopPropagation(); // Prevent triggering spot selection
    onDeleteChampion(championIndex);
  };

  return (
    <div
      className={clsx("flex flex-col h-full rounded w-[calc(100%+20px)]", {
        "bg-blue-900": is_first_team,
        "bg-red-900": !is_first_team,
      })}
    >
      {/*  very small margin bottom to because on hover the icon grows and can be cut off */}
      <div className="flex flex-col flex-1 mb-1">
        <TeamTitle is_blue_side={is_first_team} />
        <ul className="flex flex-col flex-1 justify-between mt-1">
          {roles.map((role, index) => {
            const championIndex = index as ChampionIndex;
            const teamMember = team[championIndex];
            const isSelected =
              selectedSpot?.championIndex === championIndex &&
              selectedSpot?.teamIndex === pannelTeamIndex;

            return (
              <li key={index}>
                <div
                  className={clsx(
                    "flex justify-center p-1 cursor-pointer",
                    // Base styles
                    "bg-opacity-0 rounded-lg",
                    // Selected state using team colors
                    {
                      "bg-gradient-to-r shadow-[0_0_0_2px,0_0_15px_rgba(0,0,0,0.3)]":
                        isSelected,
                      "hover:bg-white/5": !isSelected,
                      "hover:scale-110": !isSelected,
                    },
                    // Team-specific colors when selected
                    {
                      "from-blue-500/20 to-blue-600/10 shadow-blue-500":
                        isSelected && is_first_team,
                      "from-red-500/20 to-red-600/10 shadow-red-500":
                        isSelected && !is_first_team,
                    }
                  )}
                  onClick={() => handleSpotClick(championIndex)}
                  onContextMenu={
                    teamMember
                      ? (e) => handleContextMenu(e, championIndex)
                      : undefined
                  }
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
                      <div className="flex items-center justify-between gap-3">
                        {/* Control buttons with team-specific positioning */}
                        <div
                          className={clsx("flex flex-col gap-2", {
                            "order-first": is_first_team,
                            "order-last": !is_first_team,
                          })}
                        >
                          {/* Lock/unlock button */}
                          <button
                            className={clsx(
                              "p-1.5 rounded transition-colors w-8 h-8 flex items-center justify-center",
                              {
                                "bg-blue-400 hover:bg-blue-300": is_first_team,
                                "bg-red-400 hover:bg-red-300": !is_first_team,
                              }
                            )}
                            onClick={(e) =>
                              toggleManualPlacement(e, championIndex)
                            }
                            title={
                              teamMember?.isManuallyPlaced
                                ? "Unlock position (allow auto-placement)"
                                : "Lock position (prevent auto-placement)"
                            }
                          >
                            {teamMember?.isManuallyPlaced ? (
                              <LockClosedIcon className="text-white w-5 h-5" />
                            ) : (
                              <LockOpenIcon className="text-white w-5 h-5" />
                            )}
                          </button>

                          {/* Delete button */}
                          <button
                            className={clsx(
                              "p-1.5 rounded transition-colors w-8 h-8 flex items-center justify-center",
                              {
                                "bg-blue-400 hover:bg-blue-300": is_first_team,
                                "bg-red-400 hover:bg-red-300": !is_first_team,
                              }
                            )}
                            onClick={(e) => handleDelete(e, championIndex)}
                            title="Remove champion"
                          >
                            <Trash2 size={20} className="text-white" />
                          </button>
                        </div>

                        {/* Champion icon */}
                        <ImageComponent
                          src={`/icons/champions/${teamMember.icon}`}
                          alt={teamMember.name}
                          className="block"
                          width={80}
                          height={80}
                        />
                      </div>
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
