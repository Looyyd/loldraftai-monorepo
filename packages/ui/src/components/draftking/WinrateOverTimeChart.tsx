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
}

export const WinrateOverTimeChart: React.FC<WinrateOverTimeChartProps> = ({
  timeBucketedPredictions,
}) => {
  const timeIntervals = [
    "0-20 min",
    "20-25 min",
    "25-30 min",
    "30-35 min",
    "35+ min",
  ];
  const timeLabels = timeIntervals;

  // Convert predictions to percentages and ensure they're in the correct order
  const winrateValues = [
    (timeBucketedPredictions["win_prediction_0_20"] as number) * 100,
    (timeBucketedPredictions["win_prediction_20_25"] as number) * 100,
    (timeBucketedPredictions["win_prediction_25_30"] as number) * 100,
    (timeBucketedPredictions["win_prediction_30_35"] as number) * 100,
    (timeBucketedPredictions["win_prediction_35_inf"] as number) * 100,
  ];

  // Create datasets for the chart
  const data = {
    labels: timeLabels,
    datasets: [
      {
        label: "Blue Team Winrate",
        data: winrateValues,
        borderColor: "rgb(59, 130, 246)", // Blue color
        backgroundColor: "rgba(59, 130, 246, 0.1)", // Light blue for fill
        tension: 0.4, // Slightly curved lines
        fill: true, // Fill area under the line
        pointRadius: 4,
        pointHoverRadius: 6,
      },
      {
        label: "Red Team Winrate",
        data: winrateValues.map((value) => 100 - value),
        borderColor: "rgb(239, 68, 68)", // Red color
        backgroundColor: "rgba(239, 68, 68, 0.1)", // Light red for fill
        tension: 0.4, // Slightly curved lines
        fill: true, // Fill area under the line
        pointRadius: 4,
        pointHoverRadius: 6,
      },
      {
        label: "50% Line",
        data: Array(timeLabels.length).fill(50),
        borderColor: "rgba(156, 163, 175, 0.5)", // Gray color
        borderWidth: 1,
        borderDash: [5, 5], // Dashed line
        pointRadius: 0, // No points
        fill: false, // No fill
        showLine: true, // Show the line
        hidden: false, // Don't hide the line
        skipNull: true, // Skip null values
        spanGaps: true, // Connect across gaps
        // Hide this dataset from tooltips
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
        display: false, // Hide legend
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
        min: 0,
        max: 100,
        ticks: {
          callback: function (value) {
            return value + "%";
          },
        },
        grid: {
          color: "rgba(156, 163, 175, 0.1)", // Light gray grid lines
        },
      },
      x: {
        grid: {
          display: false, // Hide vertical grid lines
        },
      },
    },
    interaction: {
      intersect: false,
      mode: "index",
    },
  };

  return (
    <div className="w-full h-64 p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
      <h3 className="text-center text-lg font-medium mb-2">
        Predicted Winrate Over Time
      </h3>
      <div className="h-48">
        <Line data={data} options={options} />
      </div>
    </div>
  );
};
