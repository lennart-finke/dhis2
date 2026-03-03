"""Match woreda names across DHIS2, MFR, and geoJSON/Shapefile admin_bounds."""

from pathlib import Path
import pandas as pd
import json
import re
import geopandas as gpd


# Normalization (reuse zone-level logic)
def norm(s):
    s = str(s).lower().strip()

    # Remove content in parens like (GM), (DD), (TG)
    s = re.sub(r"\s*\(.*?\)", "", s)

    # Remove facility/PHCU markers
    s = re.sub(r"_phcu$", "", s)
    s = re.sub(r"\s+phcu$", "", s)

    # Remove administrative suffixes (run twice to handle "Town Woreda" etc)
    for _ in range(3):
        s = re.sub(r"\s+health\s+office$", "", s)
        s = re.sub(r"\s+health\s+department$", "", s)
        s = re.sub(r"\s+health\s+bureau$", "", s)
        s = re.sub(r"\s+administration$", "", s)
        s = re.sub(r"\s+town\s+administration$", "", s)
        s = re.sub(r"\s+zone$", "", s)
        s = re.sub(r"\s+town$", "", s)
        s = re.sub(r"\s+woreda$", "", s)
        s = re.sub(r"\s+wereda$", "", s)
        s = re.sub(r"\s+district$", "", s)  # Census uses "district" for woredas
        s = re.sub(r"\s+special$", "", s)
        s = re.sub(r"\s+enumeration\s+area$", "", s)
        s = re.sub(r"\s+operational$", "", s)

    s = re.sub(r"\s+city\s+admin\s+woreda$", "", s)
    s = re.sub(r"\s+city\s+administration$", "", s)
    s = re.sub(r"\s+city$", "", s)
    s = re.sub(r"\s+kentiba$", "", s)

    # Remove special characters BEFORE removing "sub" to handle hyphens
    s = re.sub(r"[^a-z\s0-9]", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Now remove "sub city" and standalone "sub" after special chars are gone
    s = re.sub(r"\s+sub\s+city$", "", s)
    s = re.sub(r"\s+subcity$", "", s)
    s = re.sub(r"\s+sub$", "", s)  # Remove trailing "sub"

    # Remove "zuria" suffix (means "rural" - often not in GeoJSON)
    # Run twice to handle "zuriya" -> "zuria" -> ""
    for _ in range(2):
        s = re.sub(r"\s+zuria$", "", s)
        s = re.sub(r"\s+zuriya$", "", s)

    # Remove facility and administrative suffixes
    s = re.sub(r"\s+town\s+adminstration$", "", s)  # Typo in DHIS2 data
    s = re.sub(r"\s+town\s+adiministration$", "", s)  # Another typo
    s = re.sub(r"\s+adminstration$", "", s)  # Typo

    replacements = {
        # Specific overrides first (longer phrases)
        "elkari": "elkare serer",
        "adagnager chako": "adagn ager chaqo",
        "adagnagere chako": "adagn ager chaqo",
        "dangla zuria": "dangila",
        "ayehu guagusa": "ayehu guwagusa",
        "fagitalekoma": "fagta lakoma",
        "guagusa shikuda": "guagusa shikudad",
        "dara otilcho": "dara otilicho",
        "shebedino": "shebe dino",
        "gambella zuria": "gambela zuria",
        "hawassa zuriya": "hawassa zuria",
        "hadero tunto zuriya": "hadero tunto",
        "adilo zuria": "adilo",
        "nebaru chilga": "chilga",
        "cetral": "central",
        "enor ener meger": "enor ener",
        "mihurna aklil": "muhur na aklil",
        "aleta chukko": "aleta chuko",
        "shire endasilase": "indasilassie",
        "tahtay adyabo": "tahtay adiyabo",
        "tahtay maychew": "tahtay mayechew",
        "tahtay qoraro": "tahtay koraro",
        "degua tembien": "degua temben",
        "kola tembien": "kola temben",
        "uba debretsehay": "uba debre tsehay",
        "east badawacho": "misrak badawacho",
        "west badawacho": "mirab badowach",
        "west badewacho": "mirab badowach",
        "siraro badewacho": "siraro badawacho",
        "east azernet berbere": "misrak azenet berbere",
        "east siltie": "misrak siltie",
        "east meskan": "misrak meskan",
        "tarcha zuriya": "tarcha zuria",
        "gura ferda": "guraferda",
        "gondar zuriya": "gonder zuria",
        "efratana gidim": "efratana gidem",
        "cetral armachiho": "masero denb",
        "west dembia": "west dembiya",
        "enarje enawga": "enarj enawga",
        "enebse sar mider": "enebse sarmder",
        "goncha worho": "goncha siso enebse",
        "hulet eju nese": "hulet ej enese",
        "shebel berenta": "shebel bernta",
        "debark zuria": "debark",
        "north achefer": "semen achefer",
        "south achefer": "debub achefer",
        "south sodo": "debub sodo",
        "north mecha": "semien mecha",
        "south mecha": "debub mecha",
        "saya deberna wayu": "seyadebirna wayu",
        "mojana wodera": "mojana wadera",
        "alicho wiriro": "alicho woriro",
        "enor ener ener": "enor ener",
        "edagahamus": "edaga hamus",
        "gulomekeda": "gulo mekeda",
        "laelay adeyabo": "laelay adiabo",
        # Addis Ababa subcities spelling variations
        "akaki kaliti": "akaki kaliti",
        "kality": "kaliti",
        "nifas": "nefas",
        "kolfe keraniyo": "kolfe keraniyo",
        "kolfe": "kolfe keraniyo",
        "yaya gulele": "gulele",
        "gojjam": "gojam",
        "wolaita": "wolayita",
        "gambella": "gambela",
        "dimma": "dima",
        "mengeshi": "mengesh",
        "hawassa": "hawasa",
        "zuriya": "zuria",
        "borricha": "boricha",
        "aroressa": "aroresa",
        "daela": "daella",
        "hawella": "hawela",
        "melga": "malga",
        "hoko": "hokko",
        "yirgacheffe": "yirgachefe",
        "jor": "jore",
        "boren": "bore",
        "kelead": "kelela",
        "qooxle": "kohle qoxle",
        "harawe": "harawo",
        "elia": "elidar",
        "tula": "tulefa",
        "gereni": "gerani",
        "dangla": "dangila",
        "wensho": "wonosho",
        "abeshge": "abeshege",
        "endegagn": "endiguagn",
        "wondogenet": "wondo genet",
        "ahferom": "aheferom",
        "welkayit": "wolkait",
        "wolkait": "welkait",
        "seharti": "saharti",
        "maekelay": "laelay",
        "qedamay": "kedamay",
        "quiha": "kuiha",
        "kaffa": "kefa",
        "dawro": "dawuro",
        "enor": "enor ener",
        "wogidi": "wegde",
        "mehal saint": "mehal sayint",
        "amhara saint": "amhara sayint",
        "sinan": "senan",
        "tehuledre": "thehulederie",
        "kallu": "kalu",
        "woreilu": "were ilu",
        "worebabo": "worebabu",
        "gozamin": "guzamn",
        "basoliben": "baso liben",
        "wenberma": "wenbera",
        "libokemkem": "libo kemkem",
        "adi arkay": "adi arqay",
        "wadila": "wedeela",
        "bassona": "basona",
        "worana": "werana",
        "ebinat": "yebinet",
        "siltie": "siltie",
        "silti": "siltie",
        "chida": "chila",
        "hawzien": "hawzen",
        "lasdhankeyre": "lasdhankayre",
        "hudhet": "hudet",
        "geralta": "geraleta",
        "kebridahar": "kebridehar",
        "adigudem": "adigudom",
        "degahbur": "degehabur",
        "degahbur town": "degahabur town",
        "afdam": "afdem",
        "erop": "erob",
        "neksege": "neqsege",
        "elele": "elale",
        "freweini": "freweyni",
        "gimbichau": "gimbi",
        "dawa harewa": "dewa harewa",
        "adiyo": "adiyio",
        "teppi": "tepi",
        "zalla": "zala",
        "demboya": "damboya",
        "kacha birra": "kacha bira",
        "shinshicho": "shinshincho",
        "motta": "mota",
        "doyo gena": "doyogena",
        "bare": "barey",
        "buee": "bule",
        "jemu": "jeju",
        "babili": "babile",
        "kawo koyisha": "kawo koisha",
        "pawi": "pawe",
        "shoarobit": "shoa robit",
        "lanfuro": "lanfero",
        "siz": "size",
        "mubarak": "mubarek",
        "gorigesha": "gori gesha",
        "mitto": "mito",
        "zaba gazo": "zabagazo",
        "bodalay": "bodaley",
        "tello": "tullo",
        "fofa": "ofa",
        "dekasuftu": "deka suftu",
        "mizhiga": "mizyiga",
        "bokolmanyo": "bokolmayo",
        "abaqaraw": "abakorow",
        "berano": "berocano",
        "danbal": "dembel",
        "dhanan": "danan",
        "dhoboweyn": "debeweyin",
        "dollo bay": "dolobay",
        "east imey": "east imi",
        "gorabaqaqsa": "goro baqaqsa",
        "gura damole": "guradamole",
        "hamaro": "hamero",
        "horashagah": "horshagah",
        "kadadumo": "qada duma",
        "koran mula": "koran mulla",
        "lehelyuub": "lehelyucub",
        "maeso": "miesso",
        "mayumuluqo": "meyumuluka",
        "qabribayah": "kebribayah",
        "qalafo": "kelafo",
        "shaykosh": "shaygosh",
        "togwajale": "wajale",
        "west imay": "west imi",
        "yahoob": "yahob",
        "dambe": "dembe",
        "danot": "danod",
        "machakel": "michakel",
        "bibugne": "bibugn",
        "ahsia": "ahsea",
        "setit humra": "setit humera",
        "emdibir": "emdebir",
        "asagirt": "assagirt",
        "bugina": "bugna",
        "birqod": "burqod",
        "buliki": "bulike",
        "kindo koyisha": "kindo koyesha",
        "harorays": "haroreys",
        "chareti": "charati",
        "duhun": "dihun",
        "daroor": "daror",
        "zay": "zayi",
        "jamma": "jama",
        "shabeley": "shabeeley",
        "hadagala": "hadhagala",
        "lemo": "lemmo",
        "sehala": "shala",
        "hulbareg": "wulbareg",
        "menz mama midir": "menze mama midir",
        "menz lalo midir": "menze lalo midir",
        "suri": "surma",
        "gnangatom": "nyngatom",
        "offa": "ofa",
        "sedie": "sedae",
        "sedie muja": "sede muja",
        "kochore": "kochere",
        "ziquala": "zequala",
        "kinfaze begela": "kinfaz begela",
        "arbaminch zuria": "arba minch zuria",
        "borena": "boreda",
        "beto": "bero",
        "chilga": "chila",
        "gazgibla": "gaz gibla",
        "bahir dar zuria": "bahirdar zuria",
        "guna begemidir": "guna begemider",
        "kewot": "kewet",
        "wogera": "wegera",
        "garda martha": "garda marta",
        "agena": "gena",
        "wera dijo": "wera",
        "raya chercher": "chercher",
        "sewha saesie": "saesie",
        "may kadra": "may kadra",
        "abergele yechila": "abergele",
        "laelay tselemti": "tselemti",
        "rama adiarbaete": "rama",
        "ann lemo": "lemmo",
        "ann lemmo": "lemmo",
        "alem gebeya": "ejere addis alem",
        "deri saja zuria": "bahirdar zuria",
        "adwa": "adwa",
        "wanthoa": "wantawo",
        "menz keya gebreal": "menze keya gabriel",
        "tanqua-milash": "tanqua melashe",
        "wajerat": "wajirat",
        "keyih tekli": "keyhe tekli",
        "toba": "yem sp",
        "saja": "yem sp",
        "wolkite": "welkite",
        # Mekelle subcities
        "qedamay woyane": "mekelle",
        "hadnet": "mekelle",
        "ayder": "mekelle",
        "hawelti": "mekelle",
        "semen": "mekelle",
        "adi haqi": "mekelle",
        "adaar": "adar",
        "ayssaita": "asayita",
        "mille": "mile",
        "abala": "abaala",
        "berahle": "berahile",
        "erabti": "erebti",
        "magale": "megale",
        "awura": "awra",
        "ewa": "euwa",
        "yallo": "yalo",
        "dalifaghe": "dalefage",
        "hadeleala": "hadelela",
        "telalak": "telalek",
        "menz": "menze",
        "wodera": "wadera",
        "worda": "",
        "worho": "",
        "ankasha guagusa": "ankasha",
        "gishe rabel": "gishe",
        "hobicha abaya": "hobicha",
        "abaala abaya": "abaya",
        "sedae muja": "sedae",
        "andabet west esite": "andabet",
        "loma bosa": "loma",
        "gena bosa": "gena",
        "konta koysha": "konta",
        "mojana wadera": "wadera",
        "gununo hamus": "gununo",
        "dasenech kuraz": "dasenech",
        "ezo kogota": "kogota",
        "deka suftu": "deka",
        "maji tum": "maji",
        "woba ari": "ari",
        "azernet berbere": "berbere",
        "azenet berbere": "berbere",
        "awash fenteale": "awash",
        "shey bench": "bench",
        "gide bench": "bench",
        "amhara sayint": "sayint",
        "deri yem sp": "yem sp",
        # Census-specific normalizations
        "adis abeba": "addis ababa",
        "gondar": "gonder",
        "hulla": "hula",
        "chire": "kochire",
        "aleta wondo": "aleta wendo",
        "wondo genet": "wondogenet",
        "wondo-genet": "wondogenet",
        "loka abaya": "loko abeya",
        "dara otilicho": "dara otilcho",
    }

    for k, v in replacements.items():
        # Use regex boundary to avoid partial replacement (e.g. guagusa shikuda -> guagusa shikudadd)
        s = re.sub(r"\b" + re.escape(k) + r"\b", v, s)

    # Remove directional prefixes that might not be in GeoJSON
    # Do this AFTER specific replacements to allow "north mecha" -> "semien mecha" etc
    s = re.sub(r"^north\s+", "", s)
    s = re.sub(r"^south\s+", "", s)
    s = re.sub(r"^east\s+", "", s)
    s = re.sub(r"^west\s+", "", s)
    s = re.sub(r"^semien\s+", "", s)  # Amharic for "north"
    s = re.sub(r"^debub\s+", "", s)  # Amharic for "south"
    s = re.sub(r"^misrak\s+", "", s)  # Amharic for "east"
    s = re.sub(r"^mirab\s+", "", s)  # Amharic for "west"

    # Remove duplicate consecutive words (e.g., "kolfe keraniyo keraniyo" -> "kolfe keraniyo")
    words = s.split()
    deduped = []
    for i, word in enumerate(words):
        if i == 0 or word != words[i - 1]:
            deduped.append(word)
    s = " ".join(deduped)

    # Final cleanup of extra spaces
    s = re.sub(r"\s+", " ", s).strip()

    return s


def norm_region(s):
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s+regional state$", "", s)
    s = re.sub(r"\s+regional health bureau$", "", s)
    s = re.sub(r"\s+region$", "", s)
    s = re.sub(r"\s+city administration$", "", s)
    s = re.sub(r"-", " ", s)
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\bgambella\b", "gambela", s)
    return s


# Load DHIS2 woredas
dhis_files = list(Path("data/dhis2/25_12_12").glob("*.csv"))
dhis_list = []
for f in dhis_files:
    df = pd.read_csv(
        f,
        usecols=lambda c: c.startswith("orgunit") or c.startswith("organisation"),
        low_memory=False,
    )
    dhis_list.append(df)
dhis = pd.concat(dhis_list, ignore_index=True)

# Woreda is Level 4 for most regions, but Level 3 for Addis Ababa (subcities)
# For Addis Ababa, use level 3 as woreda since subcities are stored there
is_addis = dhis["orgunitlevel2"] == "Addis Ababa City Administration"
dhis["woreda"] = dhis.get("orgunitlevel4")
dhis["zone"] = dhis.get("orgunitlevel3")
# Override for Addis Ababa: subcities are at level 3
dhis.loc[is_addis, "woreda"] = dhis.loc[is_addis, "orgunitlevel3"]
dhis.loc[is_addis, "zone"] = dhis.loc[
    is_addis, "orgunitlevel2"
]  # Use region as zone for AA

# Identify pseudo-woredas (where woreda == organisationunitname means it's a facility)
mask_facility_name = dhis["woreda"] == dhis["organisationunitname"]

# Identify facilities with _PHCU suffix or other facility markers
mask_phcu = dhis["woreda"].str.contains("_PHCU", case=False, na=False) | dhis[
    "woreda"
].str.contains(" PHCU$", case=False, na=False)

# Identify hospitals and health centers
mask_type = (
    dhis["woreda"].str.contains("Hospital", case=False, na=False)
    | dhis["woreda"].str.contains("Health Center", case=False, na=False)
    | dhis["woreda"].str.contains("Health Office$", case=False, na=False)
)

# Assign to dummy woreda
# mask_invalid = mask_facility_name | mask_phcu | mask_type
# dhis.loc[mask_invalid, "woreda"] = "Unidentified Woreda"

# Keep rows where woreda is not NA
dhis = dhis[dhis["woreda"].notna()]


dhis_woredas = dhis[["orgunitlevel2", "zone", "woreda"]].drop_duplicates()
dhis_woredas = dhis_woredas.rename(
    columns={
        "orgunitlevel2": "dhis_region",
        "zone": "dhis_zone",
        "woreda": "dhis_woreda",
    }
)
dhis_woredas["dhis_woreda_norm"] = dhis_woredas["dhis_woreda"].apply(norm)
dhis_woredas["dhis_zone_norm"] = dhis_woredas["dhis_zone"].apply(norm)

# Region mapping for DHIS2 -> MFR
region_map = {
    "Central Ethiopian region": "SNNP Region",
    "South Ethiopia": "SNNP Region",
    "South Ethiopia Regional State": "SNNP Region",
    "South Ethiopia Region": "SNNP Region",
    "South West Ethiopia Region": "SNNP Region",
    "Southern West Ethiopia Region": "SNNP Region",
    "Sidama Region": "SNNP Region",
}
dhis_woredas["dhis_region_mapped"] = dhis_woredas["dhis_region"].replace(region_map)
dhis_woredas["dhis_region_norm"] = dhis_woredas["dhis_region_mapped"].apply(norm_region)

# Load MFR woredas
regional_sheets = [
    "SWE",
    "SLI",
    "SNNP",
    "SD",
    "OR",
    "GM",
    "BG",
    "AMH",
    "AFA",
    "AA",
    "HAR",
    "DD",
]
mfr_list = []
for sheet in regional_sheets:
    df = pd.read_excel("data/mfr/MFR List All.xlsx", sheet_name=sheet)
    mfr_list.append(df)

main_sheet = pd.read_excel(
    "data/mfr/MFR List All.xlsx", sheet_name="MFR List All - Feb 23"
)
tigray = main_sheet[
    main_sheet["region"].astype(str).str.contains("Tigray", case=False, na=False)
].copy()
mfr_list.append(tigray)

mfr = pd.concat(mfr_list, ignore_index=True)
mfr_woredas = (
    mfr[["region", "zone", "woreda"]].drop_duplicates().dropna(subset=["woreda"])
)
mfr_woredas = mfr_woredas[mfr_woredas["woreda"].str.strip() != ""]
mfr_woredas["mfr_woreda_norm"] = mfr_woredas["woreda"].apply(norm)
mfr_woredas["mfr_zone_norm"] = mfr_woredas["zone"].apply(norm)
mfr_woredas["mfr_region_norm"] = mfr_woredas["region"].apply(norm_region)
mfr_woredas = mfr_woredas.rename(
    columns={"region": "mfr_region", "zone": "mfr_zone", "woreda": "mfr_woreda"}
)

# Load GeoJSON woredas (admin3)
with open("data/admin_bounds/eth_admin_boundaries.geojson/eth_admin3.geojson") as f:
    geo = json.load(f)
geo_woredas = pd.DataFrame(
    [
        {
            "geo_woreda": feat["properties"]["adm3_name"],
            "geo_pcode": feat["properties"]["adm3_pcode"],
            "geo_zone": feat["properties"]["adm2_name"],
            "geo_region": feat["properties"]["adm1_name"],
        }
        for feat in geo["features"]
    ]
)
geo_woredas["geo_woreda_norm"] = geo_woredas["geo_woreda"].apply(norm)
geo_woredas["geo_zone_norm"] = geo_woredas["geo_zone"].apply(norm)

# Map GeoJSON region names to match DHIS2 normalized regions
GEO_TO_DHIS_REGION_MAP = {
    "SNNP": "snnp",
    "South West Ethiopia": "snnp",  # SW Ethiopia is part of former SNNP
    "Sidama": "snnp",  # Sidama is part of former SNNP
    "Addis Ababa": "addis ababa",
    "Afar": "afar",
    "Amhara": "amhara",
    "Benishangul Gumz": "benishangul gumuz",
    "Dire Dawa": "dire dawa",
    "Gambela": "gambela",  # GeoJSON uses single 'l', DHIS2 'Gambella' normalizes via replacement
    "Harari": "harari",
    "Oromia": "oromia",
    "Somali": "somali",
    "Tigray": "tigray",
}
geo_woredas["geo_region_mapped"] = geo_woredas["geo_region"].map(GEO_TO_DHIS_REGION_MAP)
geo_woredas["geo_region_norm"] = geo_woredas["geo_region_mapped"].fillna(
    geo_woredas["geo_region"].apply(norm_region)
)

# Load Shapefile woredas (admin3)
shp_path = "data/geography/Ethiopia_adm3_uscb_2016.shp"
shp_gdf = gpd.read_file(shp_path)


def extract_shp_parts(geo_match):
    """Extract region, zone, woreda from GEO_MATCH like ETHIOPIA_TIGRAY_NORTH WESTERN TIGRAY_TAHTAY ADIYABO"""
    if not geo_match:
        return "", "", ""
    parts = geo_match.split("_")
    if len(parts) >= 4:
        return parts[1], parts[2], parts[3]  # region, zone, woreda
    elif len(parts) == 3:
        return parts[1], "", parts[2]
    return "", "", ""


shp_woredas = pd.DataFrame(
    {
        "shp_woreda": shp_gdf["AREA_NAME"],
        "shp_geo_match": shp_gdf["GEO_MATCH"],
    }
)
shp_parts = shp_woredas["shp_geo_match"].apply(extract_shp_parts)
shp_woredas["shp_raw_region"] = [p[0] for p in shp_parts]
shp_woredas["shp_raw_zone"] = [p[1] for p in shp_parts]
shp_woredas["shp_woreda_norm"] = shp_woredas["shp_woreda"].apply(norm)

SHP_REGION_MAP = {
    "AMARA": "amhara",
    "OROMIYA": "oromia",
    "AFAR": "afar",
    "BINSHANGUL GUMUZ": "benishangul gumuz",
    "DIRE DAWA": "dire dawa",
    "GAMBELA HIZBOCH": "gambella",
    "HARERI HIZB": "harari",
    "YEDEBUB BIHEROCH BIHERESEBOCH NA HIZBOCH": "snnp",
    "SUMALE": "somali",
    "TIGRAY": "tigray",
    "ADIS ABEBA": "addis ababa",
}
shp_woredas["shp_region_norm"] = shp_woredas["shp_raw_region"].replace(SHP_REGION_MAP)
has_shapefile = True

# Load Census data (treating subcities as woredas)
census_path = "data/census_populations/census_data.csv"
census_df = pd.read_csv(census_path)

# Filter to get only Districts, Towns, and Sub Cities (census equivalents of woredas)
census_woredas = census_df[
    census_df["status"].isin(
        [
            "District",
            "Town",
            "Sub City",
            "Special District",
            "National Park",
            "Special Census District",
        ]
    )
].copy()

census_woredas["census_woreda"] = census_woredas["name"]
census_woredas["census_woreda_norm"] = census_woredas["name"].apply(norm)
census_woredas["census_id"] = census_woredas["id"]

# Keep only relevant columns
census_woredas = census_woredas[
    ["census_woreda", "census_woreda_norm", "census_id"]
].drop_duplicates()

has_census = True
print(f"Loaded {len(census_woredas)} census woredas (districts, towns, and subcities)")

# GeoJSON -> Census matching (for reference)
if has_census:
    geo_census_match = geo_woredas.merge(
        census_woredas[["census_woreda", "census_id", "census_woreda_norm"]],
        left_on="geo_woreda_norm",
        right_on="census_woreda_norm",
        how="left",
    )
    geo_total = len(geo_woredas)
    geo_matched = geo_census_match["census_woreda"].notna().sum()
    print(
        f"GeoJSON → Census: {geo_matched}/{geo_total} ({100*geo_matched/geo_total:.1f}%)"
    )

    # Save this mapping for use in other scripts
    geo_census_save = geo_census_match[
        [
            "geo_woreda",
            "geo_pcode",
            "census_woreda",
            "census_id",
            "geo_woreda_norm",
            "census_woreda_norm",
        ]
    ].drop_duplicates()
    geo_census_save.to_csv("estimates/geojson_census_mapping.csv", index=False)
    print(
        f"Saved {len(geo_census_save)} GeoJSON-Census mappings to estimates/geojson_census_mapping.csv"
    )

    # MFR -> Census matching (for reference)
    mfr_census_match = mfr_woredas.merge(
        census_woredas[["census_woreda", "census_id", "census_woreda_norm"]],
        left_on="mfr_woreda_norm",
        right_on="census_woreda_norm",
        how="left",
    )
    mfr_total = len(mfr_woredas)
    mfr_matched = mfr_census_match["census_woreda"].notna().sum()
    print(
        f"MFR → Census:     {mfr_matched}/{mfr_total} ({100*mfr_matched/mfr_total:.1f}%)"
    )

# Three-way matching: DHIS2 -> MFR (by woreda + zone + region)
mapping = dhis_woredas.merge(
    mfr_woredas,
    left_on=["dhis_woreda_norm", "dhis_zone_norm", "dhis_region_norm"],
    right_on=["mfr_woreda_norm", "mfr_zone_norm", "mfr_region_norm"],
    how="left",
)
# DHIS2 -> GeoJSON (by woreda norm AND region to avoid cross-region false matches)
mapping = mapping.merge(
    geo_woredas[
        [
            "geo_woreda",
            "geo_pcode",
            "geo_zone",
            "geo_region",
            "geo_woreda_norm",
            "geo_region_norm",
        ]
    ],
    left_on=["dhis_woreda_norm", "dhis_region_norm"],
    right_on=["geo_woreda_norm", "geo_region_norm"],
    how="left",
)

# DHIS2 -> Shapefile
mapping = mapping.merge(
    shp_woredas[["shp_woreda", "shp_geo_match", "shp_woreda_norm"]],
    left_on="dhis_woreda_norm",
    right_on="shp_woreda_norm",
    how="left",
)

# DHIS2 -> Census
mapping = mapping.merge(
    census_woredas[["census_woreda", "census_id", "census_woreda_norm"]],
    left_on="dhis_woreda_norm",
    right_on="census_woreda_norm",
    how="left",
)

# Format output
mapping = mapping.rename(
    columns={
        "dhis_region": "region",
        "dhis_zone": "zone",
        "dhis_woreda": "woreda_dhis2",
        "mfr_woreda": "woreda_mfr",
        "geo_woreda": "woreda_geojson",
        "geo_pcode": "woreda_geojson_pcode",
        "shp_woreda": "woreda_shapefile",
        "shp_geo_match": "woreda_shapefile_geouid",
        "census_woreda": "woreda_census",
        "census_id": "woreda_census_id",
    }
)
output_cols = [
    "region",
    "zone",
    "woreda_dhis2",
    "woreda_mfr",
    "woreda_geojson",
    "woreda_geojson_pcode",
    "woreda_shapefile",
    "woreda_shapefile_geouid",
    "woreda_census",
    "woreda_census_id",
]
mapping = mapping[[c for c in output_cols if c in mapping.columns]].drop_duplicates()
mapping = mapping[mapping["woreda_dhis2"].notna()]

# Save
mapping.to_csv("estimates/woreda_three_way_mapping.csv", index=False)

# Statistics
total = mapping["woreda_dhis2"].nunique()
matched_mfr = mapping[mapping["woreda_mfr"].notna()]["woreda_dhis2"].nunique()
matched_geo = mapping[mapping["woreda_geojson"].notna()]["woreda_dhis2"].nunique()
matched_shp = mapping[mapping["woreda_shapefile"].notna()]["woreda_dhis2"].nunique()
matched_census = mapping[mapping["woreda_census"].notna()]["woreda_dhis2"].nunique()

print(f"\nSaved {len(mapping)} mappings to estimates/woreda_three_way_mapping.csv")
print("\nWoredas by source:")
print(f"  DHIS2 Unique: {total}")
print("\nMatch percentages (DHIS2 based):")

# DHIS2 → MFR with many-to-one
mfr_matched_data = mapping[mapping["woreda_mfr"].notna()][
    ["woreda_dhis2", "woreda_mfr"]
].drop_duplicates()
mfr_many_to_one = 0
mfr_total_in_many = 0
if len(mfr_matched_data) > 0:
    mfr_value_counts = mfr_matched_data.groupby("woreda_mfr")["woreda_dhis2"].nunique()
    mfr_many_to_one = (mfr_value_counts > 1).sum()
    mfr_total_in_many = mfr_value_counts[mfr_value_counts > 1].sum()
print(
    f"  DHIS2 → MFR:       {matched_mfr}/{total} ({100*matched_mfr/total:.1f}%), {mfr_many_to_one} target have duplicates ({mfr_total_in_many} source involved)"
)

# DHIS2 → GeoJSON with many-to-one
geo_matched_data = mapping[mapping["woreda_geojson"].notna()][
    ["woreda_dhis2", "woreda_geojson"]
].drop_duplicates()
geo_many_to_one = 0
geo_total_in_many = 0
if len(geo_matched_data) > 0:
    geo_value_counts = geo_matched_data.groupby("woreda_geojson")[
        "woreda_dhis2"
    ].nunique()
    geo_many_to_one = (geo_value_counts > 1).sum()
    geo_total_in_many = geo_value_counts[geo_value_counts > 1].sum()
print(
    f"  DHIS2 → GeoJSON:   {matched_geo}/{total} ({100*matched_geo/total:.1f}%), {geo_many_to_one} target have duplicates ({geo_total_in_many} source involved)"
)

# DHIS2 → Shapefile with many-to-one
shp_matched_data = mapping[mapping["woreda_shapefile"].notna()][
    ["woreda_dhis2", "woreda_shapefile"]
].drop_duplicates()
shp_many_to_one = 0
shp_total_in_many = 0
if len(shp_matched_data) > 0:
    shp_value_counts = shp_matched_data.groupby("woreda_shapefile")[
        "woreda_dhis2"
    ].nunique()
    shp_many_to_one = (shp_value_counts > 1).sum()
    shp_total_in_many = shp_value_counts[shp_value_counts > 1].sum()
print(
    f"  DHIS2 → Shapefile: {matched_shp}/{total} ({100*matched_shp/total:.1f}%), {shp_many_to_one} target have duplicates ({shp_total_in_many} source involved)"
)

# DHIS2 → Census with many-to-one
census_matched_data = mapping[mapping["woreda_census"].notna()][
    ["woreda_dhis2", "woreda_census"]
].drop_duplicates()
census_many_to_one = 0
census_total_in_many = 0
if len(census_matched_data) > 0:
    census_value_counts = census_matched_data.groupby("woreda_census")[
        "woreda_dhis2"
    ].nunique()
    census_many_to_one = (census_value_counts > 1).sum()
    census_total_in_many = census_value_counts[census_value_counts > 1].sum()
print(
    f"  DHIS2 → Census:    {matched_census}/{total} ({100*matched_census/total:.1f}%), {census_many_to_one} target have duplicates ({census_total_in_many} source involved)"
)

# Create separate GeoJSON → Census mapping (not filtered by DHIS2)
# This is used for population calibration in compute_total_populations.py
geo_census_mapping = geo_woredas.merge(
    census_woredas[["census_woreda", "census_id", "census_woreda_norm"]],
    left_on="geo_woreda_norm",
    right_on="census_woreda_norm",
    how="left",
)

# Keep only matched pairs
geo_census_mapping = geo_census_mapping[geo_census_mapping["census_woreda"].notna()][
    ["geo_woreda", "geo_pcode", "census_woreda", "census_id"]
].drop_duplicates()

# Save to separate file
geo_census_mapping.to_csv("estimates/geojson_census_mapping.csv", index=False)

geo_total = len(geo_woredas)
geo_matched = len(geo_census_mapping["geo_woreda"].unique())

# GeoJSON → Census many-to-one
geo_census_many_to_one = 0
geo_census_total_in_many = 0
if len(geo_census_mapping) > 0:
    geo_census_deduped = geo_census_mapping[
        ["geo_woreda", "census_woreda"]
    ].drop_duplicates()
    geo_census_value_counts = geo_census_deduped.groupby("census_woreda")[
        "geo_woreda"
    ].nunique()
    geo_census_many_to_one = (geo_census_value_counts > 1).sum()
    geo_census_total_in_many = geo_census_value_counts[
        geo_census_value_counts > 1
    ].sum()

print(
    f"\nSaved {len(geo_census_mapping)} GeoJSON→Census mappings to estimates/geojson_census_mapping.csv"
)
print(
    f"Unique GeoJSON woredas matched to census: {geo_matched}/{geo_total} ({100*geo_matched/geo_total:.1f}%), {geo_census_many_to_one} target have duplicates ({geo_census_total_in_many} source involved)"
)
