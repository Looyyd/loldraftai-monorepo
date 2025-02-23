from enum import Enum


class Champion(Enum):
    ANNIE = (1, "Annie")
    OLAF = (2, "Olaf")
    GALIO = (3, "Galio")
    TWISTED_FATE = (4, "Twisted Fate")
    XIN_ZHAO = (5, "Xin Zhao")
    URGOT = (6, "Urgot")
    LEBLANC = (7, "LeBlanc")
    VLADIMIR = (8, "Vladimir")
    FIDDLESTICKS = (9, "Fiddlesticks")
    KAYLE = (10, "Kayle")
    MASTER_YI = (11, "Master Yi")
    ALISTAR = (12, "Alistar")
    RYZE = (13, "Ryze")
    SION = (14, "Sion")
    SIVIR = (15, "Sivir")
    SORAKA = (16, "Soraka")
    TEEMO = (17, "Teemo")
    TRISTANA = (18, "Tristana")
    WARWICK = (19, "Warwick")
    NUNU = (20, "Nunu & Willump")
    MISS_FORTUNE = (21, "Miss Fortune")
    ASHE = (22, "Ashe")
    TRYNDAMERE = (23, "Tryndamere")
    JAX = (24, "Jax")
    MORGANA = (25, "Morgana")
    ZILEAN = (26, "Zilean")
    SINGED = (27, "Singed")
    EVELYNN = (28, "Evelynn")
    TWITCH = (29, "Twitch")
    KARTHUS = (30, "Karthus")
    CHOGATH = (31, "Cho'Gath")
    AMUMU = (32, "Amumu")
    RAMMUS = (33, "Rammus")
    ANIVIA = (34, "Anivia")
    SHACO = (35, "Shaco")
    DR_MUNDO = (36, "Dr. Mundo")
    SONA = (37, "Sona")
    KASSADIN = (38, "Kassadin")
    IRELIA = (39, "Irelia")
    JANNA = (40, "Janna")
    GANGPLANK = (41, "Gangplank")
    CORKI = (42, "Corki")
    KARMA = (43, "Karma")
    TARIC = (44, "Taric")
    VEIGAR = (45, "Veigar")
    TRUNDLE = (48, "Trundle")
    SWAIN = (50, "Swain")
    CAITLYN = (51, "Caitlyn")
    BLITZCRANK = (53, "Blitzcrank")
    MALPHITE = (54, "Malphite")
    KATARINA = (55, "Katarina")
    NOCTURNE = (56, "Nocturne")
    MAOKAI = (57, "Maokai")
    RENEKTON = (58, "Renekton")
    JARVAN_IV = (59, "Jarvan IV")
    ELISE = (60, "Elise")
    ORIANNA = (61, "Orianna")
    WUKONG = (62, "Wukong")
    BRAND = (63, "Brand")
    LEE_SIN = (64, "Lee Sin")
    VAYNE = (67, "Vayne")
    RUMBLE = (68, "Rumble")
    CASSIOPEIA = (69, "Cassiopeia")
    SKARNER = (72, "Skarner")
    HEIMERDINGER = (74, "Heimerdinger")
    NASUS = (75, "Nasus")
    NIDALEE = (76, "Nidalee")
    UDYR = (77, "Udyr")
    POPPY = (78, "Poppy")
    GRAGAS = (79, "Gragas")
    PANTHEON = (80, "Pantheon")
    EZREAL = (81, "Ezreal")
    MORDEKAISER = (82, "Mordekaiser")
    YORICK = (83, "Yorick")
    AKALI = (84, "Akali")
    KENNEN = (85, "Kennen")
    GAREN = (86, "Garen")
    LEONA = (89, "Leona")
    MALZAHAR = (90, "Malzahar")
    TALON = (91, "Talon")
    RIVEN = (92, "Riven")
    KOG_MAW = (96, "Kog'Maw")
    SHEN = (98, "Shen")
    LUX = (99, "Lux")
    XERATH = (101, "Xerath")
    SHYVANA = (102, "Shyvana")
    AHRI = (103, "Ahri")
    GRAVES = (104, "Graves")
    FIZZ = (105, "Fizz")
    VOLIBEAR = (106, "Volibear")
    RENGAR = (107, "Rengar")
    VARUS = (110, "Varus")
    NAUTILUS = (111, "Nautilus")
    VIKTOR = (112, "Viktor")
    SEJUANI = (113, "Sejuani")
    FIORA = (114, "Fiora")
    ZIGGS = (115, "Ziggs")
    LULU = (117, "Lulu")
    DRAVEN = (119, "Draven")
    HECARIM = (120, "Hecarim")
    KHA_ZIX = (121, "Kha'Zix")
    DARIUS = (122, "Darius")
    JAYCE = (126, "Jayce")
    LISSANDRA = (127, "Lissandra")
    DIANA = (131, "Diana")
    QUINN = (133, "Quinn")
    SYNDRA = (134, "Syndra")
    AURELION_SOL = (136, "Aurelion Sol")
    KAYN = (141, "Kayn")
    ZOE = (142, "Zoe")
    ZYRA = (143, "Zyra")
    KAI_SA = (145, "Kai'Sa")
    SERAPHINE = (147, "Seraphine")
    GNAR = (150, "Gnar")
    ZAC = (154, "Zac")
    YASUO = (157, "Yasuo")
    VEL_KOZ = (161, "Vel'Koz")
    TALIYAH = (163, "Taliyah")
    CAMILLE = (164, "Camille")
    AKSHAN = (166, "Akshan")
    BRAUM = (201, "Braum")
    JHIN = (202, "Jhin")
    KINDRED = (203, "Kindred")
    JINX = (222, "Jinx")
    TAHM_KENCH = (223, "Tahm Kench")
    LUCIAN = (236, "Lucian")
    ZED = (238, "Zed")
    KLED = (240, "Kled")
    EKKO = (245, "Ekko")
    QIYANA = (246, "Qiyana")
    VI = (254, "Vi")
    AATROX = (266, "Aatrox")
    NAMI = (267, "Nami")
    AZIR = (268, "Azir")
    YUUMI = (350, "Yuumi")
    SAMIRA = (360, "Samira")
    THRESH = (412, "Thresh")
    ILLAOI = (420, "Illaoi")
    REK_SAI = (421, "Rek'Sai")
    IVERN = (427, "Ivern")
    KALISTA = (429, "Kalista")
    BARD = (432, "Bard")
    RAKAN = (497, "Rakan")
    XAYAH = (498, "Xayah")
    ORNN = (516, "Ornn")
    SYLAS = (517, "Sylas")
    NEEKO = (518, "Neeko")
    APHELIOS = (523, "Aphelios")
    RELL = (526, "Rell")
    PYKE = (555, "Pyke")
    VIEGO = (234, "Viego")
    SENNA = (235, "Senna")
    YONE = (777, "Yone")
    LILLIA = (876, "Lillia")
    GWEN = (887, "Gwen")
    RENATA_GLASC = (888, "Renata Glasc")
    BEL_VETH = (200, "Bel'Veth")
    ZERI = (221, "Zeri")
    NILAH = (895, "Nilah")
    K_SANTE = (897, "K'Sante")
    MILIO = (902, "Milio")
    NAAFIRI = (950, "Naafiri")
    SETT = (875, "Sett")
    BRIAR = (233, "Briar")
    HWEI = (910, "Hwei")
    SMOLDER = (901, "Smolder")
    AURORA = (893, "Aurora")
    AMBESSA = (799, "Ambessa")
    VEX = (711, "Vex")
    MEL = (800, "Mel")

    def __init__(self, id: int, display_name: str):
        self.id = id
        self.display_name = display_name


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
