import type { Team } from "@draftking/ui/lib/types";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "../ui/tooltip";
import { WinrateBar } from "./WinrateBar";
import { WinrateOverTimeChart } from "./WinrateOverTimeChart";
import type { ImageComponent } from "@draftking/ui/lib/types";

interface DraftAnalysisShowcaseProps {
  prediction: {
    win_probability: number;
    gold_diff_15min: number[];
    champion_impact: number[];
    time_bucketed_predictions: Record<string, number>;
  };
  team1: Team;
  team2: Team;
  ImageComponent: ImageComponent; // Add ImageComponent prop
}

export const DraftAnalysisShowcase = ({
  prediction,
  team1,
  team2,
  ImageComponent,
}: DraftAnalysisShowcaseProps) => {
  const { win_probability, gold_diff_15min, champion_impact, time_bucketed_predictions } = prediction;

  return (
    <TooltipProvider delayDuration={0}>
      <div className="flex w-full flex-col items-center space-y-4">
        <div className="w-full">
          <h2 className="text-center text-lg font-bold">Predicted Winrate</h2>
          <div className="mb-1 flex w-full justify-between">
            <p>{`${win_probability.toFixed(1)}% Blue Side`}</p>
            <p>{`Red Side ${(100 - win_probability).toFixed(1)}%`}</p>
          </div>
          <WinrateBar team1Winrate={win_probability} />
        </div>

        {/* Winrate Over Time Chart */}
        <div className="w-full">
          <WinrateOverTimeChart timeBucketedPredictions={time_bucketed_predictions} />
        </div>

        <div className="w-full overflow-x-auto rounded-lg bg-white dark:bg-gray-900 shadow-sm">
          <table className="w-full min-w-[600px] table-auto">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 text-sm text-white">
                <th className="p-3 text-left font-medium">BLUE SIDE</th>
                <th className="p-2 text-right">
                  <Tooltip>
                    <TooltipTrigger className="cursor-help">
                      IMPACT
                    </TooltipTrigger>
                    <TooltipContent className="max-w-[250px] text-center whitespace-normal space-y-2">
                      <p>
                        Shows how a champion changes your team's win chance.
                        Calculated as the difference between the win rate with
                        the champion and without them
                      </p>
                      <div className="text-sm text-blue-600 dark:text-blue-400 font-medium">
                        Pro tip: Play around champions with high impact scores -
                        they are your win conditions!
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </th>
                <th className="p-2 text-center">
                  <Tooltip>
                    <TooltipTrigger className="cursor-help">
                      G@15 LEAD
                    </TooltipTrigger>
                    <TooltipContent className="max-w-[250px] text-center whitespace-normal space-y-2">
                      <p>
                        Predicted gold difference at 15 minutes against opposing
                        laner.
                      </p>
                      <div className="text-sm text-blue-600 dark:text-blue-400 font-medium">
                        Pro tip: Use this to identify winning lanes!
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </th>
                <th className="p-2 text-center">
                  <Tooltip>
                    <TooltipTrigger className="cursor-help">
                      G@15 LEAD
                    </TooltipTrigger>
                    <TooltipContent className="max-w-[250px] text-center whitespace-normal space-y-2">
                      <p>
                        Predicted gold difference at 15 minutes against opposing
                        laner.
                      </p>
                      <div className="text-sm text-blue-600 dark:text-blue-400 font-medium">
                        Pro tip: Use this to identify winning lanes!
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </th>
                <th className="p-2 text-right">
                  <Tooltip>
                    <TooltipTrigger className="cursor-help">
                      IMPACT
                    </TooltipTrigger>
                    <TooltipContent className="max-w-[250px]  text-center whitespace-normal space-y-2">
                      <p>
                        Shows how a champion changes your team's win chance.
                        Calculated as the difference between the win rate with
                        the champion and without them
                      </p>
                      <div className="text-sm text-blue-600 dark:text-blue-400 font-medium">
                        Pro tip: Play around champions with high impact scores -
                        they are your win conditions!
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </th>
                <th className="p-3 text-right font-medium">RED SIDE</th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: 5 }).map((_, i) => (
                <tr
                  key={i}
                  className="border-b border-gray-100 dark:border-gray-800"
                >
                  <td className="p-3">
                    {team1[i as 0 | 1 | 2 | 3 | 4] && (
                      <ImageComponent
                        src={`/icons/champions/${
                          team1[i as 0 | 1 | 2 | 3 | 4]?.icon
                        }`}
                        alt={team1[i as 0 | 1 | 2 | 3 | 4]?.name ?? ""}
                        width={40}
                        height={40}
                        className="inline-block "
                      />
                    )}
                  </td>
                  <td
                    className={`p-3 text-right ${
                      (champion_impact[i] as number) > 0
                        ? "text-green-600"
                        : (champion_impact[i] as number) < 0
                        ? "text-red-600"
                        : ""
                    }`}
                  >
                    {champion_impact[i] !== 0 &&
                      `(${((champion_impact[i] as number) * 100).toFixed(1)}%)`}
                  </td>
                  <td className="p-3 text-center">
                    {(gold_diff_15min[i] ?? 0) > 0
                      ? Math.abs(gold_diff_15min[i] ?? 0).toFixed(0)
                      : ""}
                  </td>
                  <td className="p-3 text-center">
                    {(gold_diff_15min[i] ?? 0) < 0
                      ? Math.abs(gold_diff_15min[i] ?? 0).toFixed(0)
                      : ""}
                  </td>
                  <td
                    className={`p-3 text-right ${
                      (champion_impact[i + 5] as number) > 0
                        ? "text-green-600"
                        : (champion_impact[i + 5] as number) < 0
                        ? "text-red-600"
                        : ""
                    }`}
                  >
                    {(champion_impact[i + 5] as number) !== 0 &&
                      `(${((champion_impact[i + 5] as number) * 100).toFixed(
                        1
                      )}%)`}
                  </td>
                  <td className="p-3 text-right">
                    {team2[i as 0 | 1 | 2 | 3 | 4] && (
                      <ImageComponent
                        src={`/icons/champions/${
                          team2[i as 0 | 1 | 2 | 3 | 4]?.icon
                        }`}
                        alt={team2[i as 0 | 1 | 2 | 3 | 4]?.name ?? ""}
                        width={40}
                        height={40}
                        className="inline-block"
                      />
                    )}
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
