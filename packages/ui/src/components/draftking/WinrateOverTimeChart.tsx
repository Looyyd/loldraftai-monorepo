import React from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ChartOptions,
} from "chart.js";
import { Line } from "react-chartjs-2";
import { HelpCircle } from "lucide-react";
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "../ui/tooltip";

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface WinrateOverTimeChartProps {
  timeBucketedPredictions: Record<string, number>;
  rawTimeBucketedPredictions: Record<string, number>;
}

export const WinrateOverTimeChart: React.FC<WinrateOverTimeChartProps> = ({
  timeBucketedPredictions,
  rawTimeBucketedPredictions,
}) => {
  const [isNormalized, setIsNormalized] = React.useState(true);
  const timeIntervals = ["0-25 min", "25-30 min", "30-35 min", "35+ min"];
  const timeLabels = timeIntervals;

  // Use normalized or raw predictions based on toggle
  const predictions = isNormalized
    ? timeBucketedPredictions
    : rawTimeBucketedPredictions;

  // Convert predictions to percentages and ensure they're in the correct order
  const winrateValues = [
    (predictions["win_prediction_0_25"] as number) * 100,
    (predictions["win_prediction_25_30"] as number) * 100,
    (predictions["win_prediction_30_35"] as number) * 100,
    (predictions["win_prediction_35_inf"] as number) * 100,
  ];

  // Create datasets for the chart
  const data = {
    labels: timeLabels,
    datasets: [
      {
        label: "Blue Team Winrate",
        data: winrateValues,
        borderColor: "hsl(217, 91%, 60%)", // Using your primary color
        backgroundColor: "hsla(217, 91%, 60%, 0.1)", // Semi-transparent primary
        tension: 0.4,
        fill: false,
        pointRadius: 4,
        pointHoverRadius: 6,
      },
      {
        label: "Red Team Winrate",
        data: winrateValues.map((value) => 100 - value),
        borderColor: "hsl(0, 63%, 31%)", // Using your destructive color
        backgroundColor: "hsla(0, 63%, 31%, 0.1)", // Semi-transparent destructive
        tension: 0.4,
        fill: false,
        pointRadius: 4,
        pointHoverRadius: 6,
      },
      {
        label: "50% Line",
        data: Array(timeLabels.length).fill(50),
        borderColor: "hsla(0, 0%, 100%, 0.8)", // Using white with higher opacity
        borderWidth: 1,
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false,
        showLine: true,
        hidden: false,
        skipNull: true,
        spanGaps: true,
        tooltip: {
          enabled: false,
        },
      },
    ],
  };

  // Chart options
  const options: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        filter: function (tooltipItem) {
          // Only show tooltips for the first two datasets (Blue and Red teams), not the white line
          return tooltipItem.datasetIndex < 2;
        },
        callbacks: {
          title: function (context) {
            // Return the time interval instead of the label
            if (context[0] === undefined) {
              return undefined;
            }
            return timeIntervals[context[0].dataIndex];
          },
          label: function (context) {
            if (context.dataset.label === "50% Line") {
              return undefined;
            }
            const label = context.dataset.label || "";
            const value = context.parsed.y.toFixed(1);
            return `${label}: ${value}%`;
          },
        },
      },
    },
    scales: {
      y: {
        // Find min and max values from both blue and red team winrates
        min: Math.floor(
          Math.min(...winrateValues, ...winrateValues.map((v) => 100 - v)) - 5
        ),
        max: Math.ceil(
          Math.max(...winrateValues, ...winrateValues.map((v) => 100 - v)) + 5
        ),
        ticks: {
          callback: function (value) {
            return value + "%";
          },
          color: "hsl(213, 31%, 91%)",
        },
        grid: {
          color: "hsla(216, 34%, 17%, 0.4)",
        },
      },
      x: {
        grid: {
          display: false,
        },
        ticks: {
          color: "hsl(213, 31%, 91%)", // Using your foreground color
        },
      },
    },
    interaction: {
      intersect: false,
      mode: "index",
    },
  };

  return (
    <div className="w-full h-64 p-4 bg-card rounded-lg shadow-sm">
      <div className="flex-col space-y-2 mb-2">
        <div className="flex items-center gap-2">
          <h3 className="text-lg font-medium text-card-foreground">
            Predicted Winrate Over Time
          </h3>
          <TooltipProvider delayDuration={0}>
            <UITooltip>
              <TooltipTrigger>
                <HelpCircle className="h-5 w-5 text-gray-500" />
              </TooltipTrigger>
              <TooltipContent className="max-w-[350px] whitespace-normal">
                <p className="mb-2">
                  This chart shows how likely each team is to win IF the game
                  ends during each time period. Each prediction is specific to
                  that time window.
                </p>
                <p className="mb-2">
                  <strong>Important note:</strong> These percentages are
                  "what-if" scenarios - they show the winrate assuming the game
                  ends in that time period(surrender included). Since games are
                  more likely to end at certain times than others, you can't
                  average these numbers to get the overall winrate.
                </p>
                <p className="mb-2">
                  For example: If you see a 60% winrate before 25 minutes, that
                  means the team would win 60% of games that end that early -
                  but very few games actually end this quickly!
                </p>
                <p className="mt-2 text-sm text-blue-600 dark:text-blue-400 font-medium">
                  Pro tip: Use these predictions to understand your team's power
                  spikes!
                </p>
              </TooltipContent>
            </UITooltip>
          </TooltipProvider>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={isNormalized}
              onChange={(e) => setIsNormalized(e.target.checked)}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            Side normalize winrate (recommended)
          </label>
          <TooltipProvider delayDuration={0}>
            <UITooltip>
              <TooltipTrigger>
                <HelpCircle className="h-4 w-4 text-gray-500" />
              </TooltipTrigger>
              <TooltipContent className="max-w-[350px] whitespace-normal">
                <p className="mb-2">
                  Statistically, Blue side tends to{" "}
                  <a
                    href="https://www.leagueofgraphs.com/stats/blue-vs-red/short"
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ textDecoration: "underline" }}
                  >
                    win games earlier
                  </a>
                  , while Red side{" "}
                  <a
                    href="https://www.leagueofgraphs.com/stats/blue-vs-red/long"
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ textDecoration: "underline" }}
                  >
                    performs better in longer games.
                  </a>
                </p>
                <p className="mb-2">
                  To remove this bias, we look at the prediction from both
                  sides. For example:
                  <br />
                  • Original side: Blue side 60%
                  <br />
                  • With sides swapped: Red side 55%
                  <br />• Normalized result: (60% + 55%) ÷ 2 = 57.5%
                </p>
                <p className="text-sm text-blue-600 dark:text-blue-400 font-medium">
                  This gives you a more accurate view of your team composition's
                  power spikes, regardless of side!
                </p>
              </TooltipContent>
            </UITooltip>
          </TooltipProvider>
        </div>
      </div>
      <div className="h-48">
        <Line data={data} options={options} />
      </div>
    </div>
  );
};
