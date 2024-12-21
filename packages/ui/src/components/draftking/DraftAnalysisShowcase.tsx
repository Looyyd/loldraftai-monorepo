import React from "react";
import type { Team } from "@draftking/ui/lib/types";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "../ui/tooltip";

interface DraftAnalysisShowcaseProps {
  prediction: {
    win_probability: number;
    gold_diff_15min: number[];
    champion_impact: number[];
  };
  team1: Team;
  team2: Team;
  WinrateBar: React.ComponentType<{ team1Winrate: number }>;
}

const HeaderTooltip = ({
  children,
  content,
}: {
  children: React.ReactNode;
  content: string;
}) => (
  <th className="p-2 text-right">
    <Tooltip>
      <TooltipTrigger className="cursor-help">{children}</TooltipTrigger>
      <TooltipContent>{content}</TooltipContent>
    </Tooltip>
  </th>
);

export const DraftAnalysisShowcase = ({
  prediction,
  team1,
  team2,
  WinrateBar,
}: DraftAnalysisShowcaseProps) => {
  const { win_probability, gold_diff_15min, champion_impact } = prediction;

  return (
    <TooltipProvider delayDuration={0}>
      <div className="flex w-full flex-col items-center space-y-4">
        <div className="w-full">
          <h2 className="text-center text-lg font-bold">Predicted Winrate</h2>
          <div className="mb-1 flex w-full justify-between">
            <p>{`${Math.round(win_probability)}% Blue Side`}</p>
            <p>{`Red Side ${Math.round(100 - win_probability)}%`}</p>
          </div>
          <WinrateBar team1Winrate={win_probability} />
        </div>

        <div className="w-full overflow-x-auto">
          <table className="w-full min-w-[600px] table-auto">
            <thead>
              <tr className="border-b text-sm text-blue-600">
                <th className="p-2 text-left">BLUE SIDE</th>
                <HeaderTooltip content="How much this champion contributes to their team's win probability">
                  IMPACT
                </HeaderTooltip>
                <HeaderTooltip content="Predicted gold lead for this champion at 15 minutes">
                  G@15 LEAD
                </HeaderTooltip>
                <HeaderTooltip content="Predicted gold lead for this champion at 15 minutes">
                  G@15 LEAD
                </HeaderTooltip>
                <HeaderTooltip content="How much this champion contributes to their team's win probability">
                  IMPACT
                </HeaderTooltip>
                <th className="p-2 text-right">RED SIDE</th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: 5 }).map((_, i) => (
                <tr key={i} className="border-b">
                  <td className="p-2">
                    {team1[i as 0 | 1 | 2 | 3 | 4]?.name ?? ""}
                  </td>
                  <td
                    className={`p-2 text-right ${
                      champion_impact[i] ?? 0 > 0 ? "text-green-600" : "text-red-600"
                    }`}
                  >
                    ({(champion_impact[i] ?? 0 * 100).toFixed(1)}%)
                  </td>
                  <td className="p-2 text-right">
                    {gold_diff_15min[i] ?? 0 > 0
                      ? Math.abs(gold_diff_15min[i] ?? 0).toFixed(0)
                      : ""}
                  </td>
                  <td className="p-2 text-right">
                    {gold_diff_15min[i] ?? 0 < 0
                      ? Math.abs(gold_diff_15min[i] ?? 0).toFixed(0)
                      : ""}
                  </td>
                  <td
                    className={`p-2 text-right ${
                      champion_impact[i + 5] ?? 0 > 0
                        ? "text-green-600"
                        : "text-red-600"
                    }`}
                  >
                    ({(champion_impact[i + 5] ?? 0 * 100).toFixed(1)}%)
                  </td>
                  <td className="p-2 text-right">
                    {team2[i as 0 | 1 | 2 | 3 | 4]?.name ?? ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </TooltipProvider>
  );
}; 