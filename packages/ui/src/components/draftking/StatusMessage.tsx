import { ReactNode } from "react";
import {
  getNextPickingTeam,
  DRAFT_ORDERS,
  type DraftOrderKey,
} from "../../lib/draftLogic";
import type { Team, SelectedSpot } from "../../lib/types";

export function StatusMessage({
  selectedSpot,
  teamOne,
  teamTwo,
  selectedDraftOrder,
}: {
  selectedSpot: SelectedSpot | null;
  teamOne: Team;
  teamTwo: Team;
  selectedDraftOrder: DraftOrderKey;
}): ReactNode {
  if (selectedSpot) {
    const team = selectedSpot.teamIndex === 1 ? "BLUE" : "RED";
    const teamClass =
      selectedSpot.teamIndex === 1 ? "text-blue-500" : "text-red-500";
    return (
      <span>
        Next Pick: <span className={teamClass}>{team}</span> SELECTED SPOT
      </span>
    );
  }

  const nextTeam = getNextPickingTeam(
    teamOne,
    teamTwo,
    DRAFT_ORDERS[selectedDraftOrder]
  );
  if (!nextTeam) return "Draft Complete";

  const teamClass = nextTeam === "BLUE" ? "text-blue-500" : "text-red-500";
  return (
    <span>
      Next Pick: <span className={teamClass}>{nextTeam}</span> TEAM
    </span>
  );
}
