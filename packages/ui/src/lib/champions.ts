import championPlayRates from "./config/champion_play_rates.json";
import type { Champion } from "./types";

export const roleToIndexMap: { [key: string]: number } = {
  TOP: 0,
  JUNGLE: 1,
  MIDDLE: 2,
  BOTTOM: 3,
  UTILITY: 4,
};

interface ChampionData {
  id: number;
  name: string;
}

const championsData: ChampionData[] = [
  { id: 1, name: "Annie" },
  { id: 2, name: "Olaf" },
  { id: 3, name: "Galio" },
  { id: 4, name: "Twisted Fate" },
  { id: 5, name: "Xin Zhao" },
  { id: 6, name: "Urgot" },
  { id: 7, name: "LeBlanc" },
  { id: 8, name: "Vladimir" },
  { id: 9, name: "Fiddlesticks" },
  { id: 10, name: "Kayle" },
  { id: 11, name: "Master Yi" },
  { id: 12, name: "Alistar" },
  { id: 13, name: "Ryze" },
  { id: 14, name: "Sion" },
  { id: 15, name: "Sivir" },
  { id: 16, name: "Soraka" },
  { id: 17, name: "Teemo" },
  { id: 18, name: "Tristana" },
  { id: 19, name: "Warwick" },
  { id: 20, name: "Nunu & Willump" },
  { id: 21, name: "Miss Fortune" },
  { id: 22, name: "Ashe" },
  { id: 23, name: "Tryndamere" },
  { id: 24, name: "Jax" },
  { id: 25, name: "Morgana" },
  { id: 26, name: "Zilean" },
  { id: 27, name: "Singed" },
  { id: 28, name: "Evelynn" },
  { id: 29, name: "Twitch" },
  { id: 30, name: "Karthus" },
  { id: 31, name: "Cho'Gath" },
  { id: 32, name: "Amumu" },
  { id: 33, name: "Rammus" },
  { id: 34, name: "Anivia" },
  { id: 35, name: "Shaco" },
  { id: 36, name: "Dr. Mundo" },
  { id: 37, name: "Sona" },
  { id: 38, name: "Kassadin" },
  { id: 39, name: "Irelia" },
  { id: 40, name: "Janna" },
  { id: 41, name: "Gangplank" },
  { id: 42, name: "Corki" },
  { id: 43, name: "Karma" },
  { id: 44, name: "Taric" },
  { id: 45, name: "Veigar" },
  { id: 48, name: "Trundle" },
  { id: 50, name: "Swain" },
  { id: 51, name: "Caitlyn" },
  { id: 53, name: "Blitzcrank" },
  { id: 54, name: "Malphite" },
  { id: 55, name: "Katarina" },
  { id: 56, name: "Nocturne" },
  { id: 57, name: "Maokai" },
  { id: 58, name: "Renekton" },
  { id: 59, name: "Jarvan IV" },
  { id: 60, name: "Elise" },
  { id: 61, name: "Orianna" },
  { id: 62, name: "Wukong" },
  { id: 63, name: "Brand" },
  { id: 64, name: "Lee Sin" },
  { id: 67, name: "Vayne" },
  { id: 68, name: "Rumble" },
  { id: 69, name: "Cassiopeia" },
  { id: 72, name: "Skarner" },
  { id: 74, name: "Heimerdinger" },
  { id: 75, name: "Nasus" },
  { id: 76, name: "Nidalee" },
  { id: 77, name: "Udyr" },
  { id: 78, name: "Poppy" },
  { id: 79, name: "Gragas" },
  { id: 80, name: "Pantheon" },
  { id: 81, name: "Ezreal" },
  { id: 82, name: "Mordekaiser" },
  { id: 83, name: "Yorick" },
  { id: 84, name: "Akali" },
  { id: 85, name: "Kennen" },
  { id: 86, name: "Garen" },
  { id: 89, name: "Leona" },
  { id: 90, name: "Malzahar" },
  { id: 91, name: "Talon" },
  { id: 92, name: "Riven" },
  { id: 96, name: "Kog'Maw" },
  { id: 98, name: "Shen" },
  { id: 99, name: "Lux" },
  { id: 101, name: "Xerath" },
  { id: 102, name: "Shyvana" },
  { id: 103, name: "Ahri" },
  { id: 104, name: "Graves" },
  { id: 105, name: "Fizz" },
  { id: 106, name: "Volibear" },
  { id: 107, name: "Rengar" },
  { id: 110, name: "Varus" },
  { id: 111, name: "Nautilus" },
  { id: 112, name: "Viktor" },
  { id: 113, name: "Sejuani" },
  { id: 114, name: "Fiora" },
  { id: 115, name: "Ziggs" },
  { id: 117, name: "Lulu" },
  { id: 119, name: "Draven" },
  { id: 120, name: "Hecarim" },
  { id: 121, name: "Kha'Zix" },
  { id: 122, name: "Darius" },
  { id: 126, name: "Jayce" },
  { id: 127, name: "Lissandra" },
  { id: 131, name: "Diana" },
  { id: 133, name: "Quinn" },
  { id: 134, name: "Syndra" },
  { id: 136, name: "Aurelion Sol" },
  { id: 141, name: "Kayn" },
  { id: 142, name: "Zoe" },
  { id: 143, name: "Zyra" },
  { id: 145, name: "Kai'Sa" },
  { id: 147, name: "Seraphine" },
  { id: 150, name: "Gnar" },
  { id: 154, name: "Zac" },
  { id: 157, name: "Yasuo" },
  { id: 161, name: "Vel'Koz" },
  { id: 163, name: "Taliyah" },
  { id: 164, name: "Camille" },
  { id: 166, name: "Akshan" },
  { id: 201, name: "Braum" },
  { id: 202, name: "Jhin" },
  { id: 203, name: "Kindred" },
  { id: 221, name: "Zeri" },
  { id: 222, name: "Jinx" },
  { id: 223, name: "Tahm Kench" },
  { id: 234, name: "Viego" },
  { id: 235, name: "Senna" },
  { id: 236, name: "Lucian" },
  { id: 238, name: "Zed" },
  { id: 240, name: "Kled" },
  { id: 245, name: "Ekko" },
  { id: 246, name: "Qiyana" },
  { id: 254, name: "Vi" },
  { id: 266, name: "Aatrox" },
  { id: 267, name: "Nami" },
  { id: 268, name: "Azir" },
  { id: 350, name: "Yuumi" },
  { id: 360, name: "Samira" },
  { id: 412, name: "Thresh" },
  { id: 420, name: "Illaoi" },
  { id: 421, name: "Rek'Sai" },
  { id: 427, name: "Ivern" },
  { id: 429, name: "Kalista" },
  { id: 432, name: "Bard" },
  { id: 497, name: "Rakan" },
  { id: 498, name: "Xayah" },
  { id: 516, name: "Ornn" },
  { id: 517, name: "Sylas" },
  { id: 518, name: "Neeko" },
  { id: 523, name: "Aphelios" },
  { id: 526, name: "Rell" },
  { id: 555, name: "Pyke" },
  { id: 777, name: "Yone" },
  { id: 876, name: "Lillia" },
  { id: 887, name: "Gwen" },
  { id: 888, name: "Renata Glasc" },
  { id: 895, name: "Nilah" },
  { id: 897, name: "K'Sante" },
  { id: 901, name: "Smolder" },
  { id: 902, name: "Milio" },
  { id: 950, name: "Naafiri" },
  { id: 799, name: "Ambessa" },
  { id: 893, name: "Aurora" },
  { id: 875, name: "Sett" },
  { id: 910, name: "Hwei" },
  { id: 233, name: "Briar" },
  { id: 200, name: "Bel'Veth" },
];

/**
 * Creates a Champion object.
 * @param {ChampionData} championData - The data object containing id and name.
 * @returns {Champion} - The created Champion object.
 */
export const createChampion = (championData: ChampionData): Champion => {
  const { id, name } = championData;
  const searchName = name.replace(/[^a-zA-Z]/g, "").toLowerCase();
  const icon = `${searchName}.png`;
  return { id, name, searchName, icon };
};

export const champions: Champion[] = championsData
  .sort((a, b) => a.name.localeCompare(b.name))
  .map((championData) => createChampion(championData));

export const getChampionById = (id: number): Champion | undefined => {
  return champions.find((champion) => champion.id === id);
};

interface PlayRates {
  TOP: number;
  JUNGLE: number;
  MIDDLE: number;
  BOTTOM: number;
  UTILITY: number;
}

interface PatchPlayRates {
  [championId: string]: PlayRates;
}

interface ChampionPlayRates {
  [patch: string]: PatchPlayRates;
}

const playRatesData = championPlayRates as ChampionPlayRates;

// Get sorted patches for easy access to latest data
export const sortedPatches = Object.keys(playRatesData).sort((a, b) => {
  const [aMajor, aMinor] = a.split(".").map(Number);
  const [bMajor, bMinor] = b.split(".").map(Number);
  return bMajor === aMajor
    ? (bMinor ?? 0) - (aMinor ?? 0)
    : (bMajor ?? 0) - (aMajor ?? 0);
});

// Helper function to get champion play rates for a specific patch
function getChampionPlayRates(
  championId: number,
  patch: string
): PlayRates | undefined {
  // Try to get rates for specified patch
  const patchRates = playRatesData[patch]?.[championId.toString()];
  if (patchRates) return patchRates;

  // If not found, try latest patch
  for (const availablePatch of sortedPatches) {
    const rates = playRatesData[availablePatch]?.[championId.toString()];
    if (rates) return rates;
  }

  // If still not found, return undefined
  return undefined;
}

// Process the data to build the championToRolesMap for a specific patch
export function getChampionRoles(championId: number, patch: string): string[] {
  const playRates = getChampionPlayRates(championId, patch);

  if (!playRates) {
    return ["TOP"]; // Default to TOP if no play rate data available
  }

  return Object.entries(playRates)
    .sort((a, b) => b[1] - a[1])
    .map(([role]) => role);
}
