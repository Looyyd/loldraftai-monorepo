interface WinrateBarProps {
  team1Winrate: number;
}

export const WinrateBar = ({ team1Winrate }: WinrateBarProps) => (
  <div className="flex w-full items-center">
    <div className="relative mr-1 h-2 w-full bg-red-500">
      <div
        className="absolute h-2 bg-blue-500"
        style={{ width: `${team1Winrate}%` }}
      ></div>
    </div>
  </div>
); 