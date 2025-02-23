from enum import Enum


class ChampionClass(Enum):
    MAGE = "mage"
    TANK = "tank"
    BRUISER = "bruiser"
    ASSASSIN = "assassin"
    ADC = "adc"
    ENCHANTER = "enchanter"
    UNIQUE = "unique"


class Champion(Enum):
    ANNIE = (1, "Annie", "mage")
    OLAF = (2, "Olaf", "bruiser")
    GALIO = (3, "Galio", "tank")
    TWISTED_FATE = (4, "Twisted Fate", "mage")
    XIN_ZHAO = (5, "Xin Zhao", "bruiser")
    URGOT = (6, "Urgot", "bruiser")
    LEBLANC = (7, "LeBlanc", "assassin")
    VLADIMIR = (8, "Vladimir", "mage")
    FIDDLESTICKS = (9, "Fiddlesticks", "mage")
    KAYLE = (10, "Kayle", "mage")
    MASTER_YI = (11, "Master Yi", "assassin")
    ALISTAR = (12, "Alistar", "tank")
    RYZE = (13, "Ryze", "mage")
    SION = (14, "Sion", "tank")
    SIVIR = (15, "Sivir", "adc")
    SORAKA = (16, "Soraka", "enchanter")
    TEEMO = (17, "Teemo", "mage")
    TRISTANA = (18, "Tristana", "adc")
    WARWICK = (19, "Warwick", "bruiser")
    NUNU = (20, "Nunu & Willump", "tank")
    MISS_FORTUNE = (21, "Miss Fortune", "adc")
    ASHE = (22, "Ashe", "adc")
    TRYNDAMERE = (23, "Tryndamere", "bruiser")
    JAX = (24, "Jax", "bruiser")
    MORGANA = (25, "Morgana", "mage")
    ZILEAN = (26, "Zilean", "mage")
    SINGED = (27, "Singed", "bruiser")
    EVELYNN = (28, "Evelynn", "assassin")
    TWITCH = (29, "Twitch", "adc")
    KARTHUS = (30, "Karthus", "mage")
    CHOGATH = (31, "Cho'Gath", "tank")
    AMUMU = (32, "Amumu", "tank")
    RAMMUS = (33, "Rammus", "tank")
    ANIVIA = (34, "Anivia", "mage")
    SHACO = (35, "Shaco", "assassin")
    DR_MUNDO = (36, "Dr. Mundo", "bruiser")
    SONA = (37, "Sona", "enchanter")
    KASSADIN = (38, "Kassadin", "assassin")
    IRELIA = (39, "Irelia", "bruiser")
    JANNA = (40, "Janna", "enchanter")
    GANGPLANK = (41, "Gangplank", "bruiser")
    CORKI = (42, "Corki", "adc")
    KARMA = (43, "Karma", "enchanter")
    TARIC = (44, "Taric", "enchanter")
    VEIGAR = (45, "Veigar", "mage")
    TRUNDLE = (48, "Trundle", "bruiser")
    SWAIN = (50, "Swain", "mage")
    CAITLYN = (51, "Caitlyn", "adc")
    BLITZCRANK = (53, "Blitzcrank", "tank")
    MALPHITE = (54, "Malphite", "tank")
    KATARINA = (55, "Katarina", "assassin")
    NOCTURNE = (56, "Nocturne", "assassin")
    MAOKAI = (57, "Maokai", "tank")
    RENEKTON = (58, "Renekton", "bruiser")
    JARVAN_IV = (59, "Jarvan IV", "bruiser")
    ELISE = (60, "Elise", "mage")
    ORIANNA = (61, "Orianna", "mage")
    WUKONG = (62, "Wukong", "bruiser")
    BRAND = (63, "Brand", "mage")
    LEE_SIN = (64, "Lee Sin", "bruiser")
    VAYNE = (67, "Vayne", "adc")
    RUMBLE = (68, "Rumble", "mage")
    CASSIOPEIA = (69, "Cassiopeia", "mage")
    SKARNER = (72, "Skarner", "bruiser")
    HEIMERDINGER = (74, "Heimerdinger", "mage")
    NASUS = (75, "Nasus", "bruiser")
    NIDALEE = (76, "Nidalee", "mage")
    UDYR = (77, "Udyr", "bruiser")
    POPPY = (78, "Poppy", "tank")
    GRAGAS = (79, "Gragas", "tank")
    PANTHEON = (80, "Pantheon", "assassin")
    EZREAL = (81, "Ezreal", "adc")
    MORDEKAISER = (82, "Mordekaiser", "bruiser")
    YORICK = (83, "Yorick", "bruiser")
    AKALI = (84, "Akali", "assassin")
    KENNEN = (85, "Kennen", "mage")
    GAREN = (86, "Garen", "bruiser")
    LEONA = (89, "Leona", "tank")
    MALZAHAR = (90, "Malzahar", "mage")
    TALON = (91, "Talon", "assassin")
    RIVEN = (92, "Riven", "bruiser")
    KOG_MAW = (96, "Kog'Maw", "adc")
    SHEN = (98, "Shen", "tank")
    LUX = (99, "Lux", "mage")
    XERATH = (101, "Xerath", "mage")
    SHYVANA = (102, "Shyvana", "unique")
    AHRI = (103, "Ahri", "mage")
    GRAVES = (104, "Graves", "adc")
    FIZZ = (105, "Fizz", "assassin")
    VOLIBEAR = (106, "Volibear", "bruiser")
    RENGAR = (107, "Rengar", "assassin")
    VARUS = (110, "Varus", "adc")
    NAUTILUS = (111, "Nautilus", "tank")
    VIKTOR = (112, "Viktor", "mage")
    SEJUANI = (113, "Sejuani", "tank")
    FIORA = (114, "Fiora", "bruiser")
    ZIGGS = (115, "Ziggs", "mage")
    LULU = (117, "Lulu", "enchanter")
    DRAVEN = (119, "Draven", "adc")
    HECARIM = (120, "Hecarim", "bruiser")
    KHA_ZIX = (121, "Kha'Zix", "assassin")
    DARIUS = (122, "Darius", "bruiser")
    JAYCE = (126, "Jayce", "unique")
    LISSANDRA = (127, "Lissandra", "mage")
    DIANA = (131, "Diana", "assassin")
    QUINN = (133, "Quinn", "assassin")
    SYNDRA = (134, "Syndra", "mage")
    AURELION_SOL = (136, "Aurelion Sol", "mage")
    KAYN = (141, "Kayn", "unique")
    ZOE = (142, "Zoe", "mage")
    ZYRA = (143, "Zyra", "mage")
    KAI_SA = (145, "Kai'Sa", "adc")
    SERAPHINE = (147, "Seraphine", "enchanter")
    GNAR = (150, "Gnar", "bruiser")
    ZAC = (154, "Zac", "tank")
    YASUO = (157, "Yasuo", "bruiser")
    VEL_KOZ = (161, "Vel'Koz", "mage")
    TALIYAH = (163, "Taliyah", "mage")
    CAMILLE = (164, "Camille", "bruiser")
    AKSHAN = (166, "Akshan", "assassin")
    BRAUM = (201, "Braum", "tank")
    JHIN = (202, "Jhin", "adc")
    KINDRED = (203, "Kindred", "adc")
    JINX = (222, "Jinx", "adc")
    TAHM_KENCH = (223, "Tahm Kench", "tank")
    LUCIAN = (236, "Lucian", "adc")
    ZED = (238, "Zed", "assassin")
    KLED = (240, "Kled", "bruiser")
    EKKO = (245, "Ekko", "assassin")
    QIYANA = (246, "Qiyana", "assassin")
    VI = (254, "Vi", "bruiser")
    AATROX = (266, "Aatrox", "bruiser")
    NAMI = (267, "Nami", "enchanter")
    AZIR = (268, "Azir", "mage")
    YUUMI = (350, "Yuumi", "enchanter")
    SAMIRA = (360, "Samira", "adc")
    THRESH = (412, "Thresh", "tank")
    ILLAOI = (420, "Illaoi", "bruiser")
    REK_SAI = (421, "Rek'Sai", "bruiser")
    IVERN = (427, "Ivern", "enchanter")
    KALISTA = (429, "Kalista", "adc")
    BARD = (432, "Bard", "unique")
    RAKAN = (497, "Rakan", "enchanter")
    XAYAH = (498, "Xayah", "adc")
    ORNN = (516, "Ornn", "tank")
    SYLAS = (517, "Sylas", "bruiser")
    NEEKO = (518, "Neeko", "mage")
    APHELIOS = (523, "Aphelios", "adc")
    RELL = (526, "Rell", "tank")
    PYKE = (555, "Pyke", "assassin")
    VIEGO = (234, "Viego", "assassin")
    SENNA = (235, "Senna", "unique")
    YONE = (777, "Yone", "assassin")
    LILLIA = (876, "Lillia", "mage")
    GWEN = (887, "Gwen", "bruiser")
    RENATA_GLASC = (888, "Renata Glasc", "enchanter")
    BEL_VETH = (200, "Bel'Veth", "bruiser")
    ZERI = (221, "Zeri", "adc")
    NILAH = (895, "Nilah", "adc")
    K_SANTE = (897, "K'Sante", "tank")
    MILIO = (902, "Milio", "enchanter")
    NAAFIRI = (950, "Naafiri", "assassin")
    SETT = (875, "Sett", "bruiser")
    BRIAR = (233, "Briar", "bruiser")
    HWEI = (910, "Hwei", "mage")
    SMOLDER = (901, "Smolder", "adc")
    AURORA = (893, "Aurora", "mage")
    AMBESSA = (799, "Ambessa", "assassin")
    VEX = (711, "Vex", "mage")
    MEL = (800, "Mel", "mage")

    def __init__(
        self, champion_id: int, display_name: str, champion_class: str
    ) -> None:
        self.id: int = champion_id
        self.display_name: str = display_name
        self.champion_class: ChampionClass = ChampionClass(champion_class)


# Valid ids, because they are not incremental
VALID_CHAMPION_IDS = [champion.id for champion in Champion]


# Champion role constants with names as comments
TOP_CHAMPIONS = [
    Champion.OLAF.id,
    Champion.URGOT.id,
    Champion.VLADIMIR.id,
    Champion.RYZE.id,
    Champion.KAYLE.id,
    Champion.SION.id,
    Champion.TEEMO.id,
    Champion.WARWICK.id,
    Champion.TRYNDAMERE.id,
    Champion.JAX.id,
    Champion.SINGED.id,
    Champion.CHOGATH.id,
    Champion.DR_MUNDO.id,
    Champion.IRELIA.id,
    Champion.GANGPLANK.id,
    Champion.TRUNDLE.id,
    Champion.MALPHITE.id,
]

JUNGLE_CHAMPIONS = [
    Champion.OLAF.id,
    Champion.XIN_ZHAO.id,
    Champion.FIDDLESTICKS.id,
    Champion.MASTER_YI.id,
    Champion.WARWICK.id,
    Champion.NUNU.id,
    Champion.EVELYNN.id,
    Champion.SHACO.id,
    Champion.TRUNDLE.id,
]

MID_CHAMPIONS = [
    Champion.ANNIE.id,
    Champion.GALIO.id,
    Champion.TWISTED_FATE.id,
    Champion.TRISTANA.id,
    Champion.LEBLANC.id,
    Champion.VLADIMIR.id,
    Champion.RYZE.id,
    Champion.CHOGATH.id,
    Champion.RAMMUS.id,
    Champion.KASSADIN.id,
    Champion.IRELIA.id,
    Champion.CORKI.id,
    Champion.KATARINA.id,
]

BOT_CHAMPIONS = [
    Champion.SIVIR.id,
    Champion.TRISTANA.id,
    Champion.MISS_FORTUNE.id,
    Champion.ASHE.id,
    Champion.TWITCH.id,
    Champion.KARTHUS.id,
    Champion.SWAIN.id,
    Champion.CAITLYN.id,
]

UTILITY_CHAMPIONS = [
    Champion.ALISTAR.id,
    Champion.SORAKA.id,
    Champion.MORGANA.id,
    Champion.ZILEAN.id,
    Champion.SONA.id,
    Champion.JANNA.id,
    Champion.KARMA.id,
    Champion.TARIC.id,
    Champion.BLITZCRANK.id,
]

# Combine all role-based champions into a single list
ROLE_CHAMPIONS = [
    TOP_CHAMPIONS,
    JUNGLE_CHAMPIONS,
    MID_CHAMPIONS,
    BOT_CHAMPIONS,
    UTILITY_CHAMPIONS,
]
