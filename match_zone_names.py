"""Match zone names across DHIS2, MFR, and geoJSON admin_bounds."""

from pathlib import Path
import pandas as pd
import json
import re
import geopandas as gpd


# Normalization for matching
def norm(s):
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()

    # Specific fix for Afar "Name /Zone X" -> "Zone X"
    if "/zone" in s:
        match = re.search(r"/zone\s*(\d+)", s)
        if match:
            return f"zone {match.group(1)}"

    # Specific fix for "Bahir Dar" -> "bahirdar"
    if "bahir dar" in s:
        s = s.replace("bahir dar", "bahirdar")

    s = re.sub(r"\s+health department$", "", s)
    s = re.sub(r"\s+health bureau$", "", s)
    s = re.sub(
        r"\s+administration$", "", s
    )  # Fixes "Town Administration" - Remove first!
    s = re.sub(r"\s+zone$", "", s)
    s = re.sub(r"\s+town$", "", s)
    s = re.sub(r"\s+woreda$", "", s)
    s = re.sub(r"\s+special$", "", s)
    s = re.sub(r"\s+enumeration\s+area$", "", s)
    s = re.sub(r"\s+sub\s*city$", "", s)  # Fix for "Sub City" / "Subcity"

    # Fix spellings
    s = s.replace("gojjam", "gojam")
    s = s.replace("wolaita", "wolayita")

    s = re.sub(r"[^a-z\s0-9]", "", s)  # Keep numbers for Zone 1, 2 etc
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Region normalization for matching
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
    return s


# Load DHIS2 zones
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

CITY_REGIONS = [
    "Dire Dawa City Administration"
]  # Addis Ababa uses level 3 (Sub Cities)
dhis["zone"] = dhis.apply(
    lambda r: r["orgunitlevel2"]
    if r.get("orgunitlevel2") in CITY_REGIONS
    else r.get("orgunitlevel3"),
    axis=1,
)

# Assign facilities misclassified as zones to a dummy zone
# mask = dhis["zone"] == dhis["organisationunitname"]
# dhis.loc[mask, "zone"] = "Unidentified Zone"

dhis_zones = (
    dhis[["orgunitlevel2", "zone", "organisationunitcode", "organisationunitname"]]
    .drop_duplicates()
    .dropna(subset=["zone"])
)
dhis_unique = dhis_zones[["orgunitlevel2", "zone"]].drop_duplicates()
dhis_unique["dhis_zone_norm"] = dhis_unique["zone"].apply(norm)

# Manual Region Mapping for DHIS2 -> MFR
region_map = {
    "Central Ethiopian region": "SNNP Region",
    "South Ethiopia": "SNNP Region",
    "South Ethiopia Regional State": "SNNP Region",
    "South Ethiopia Region": "SNNP Region",
    "South West Ethiopia Region": "SNNP Region",  # Map to SNNP for shapefile matching
    "Southern West Ethiopia Region": "SNNP Region",  # Alternate name
    "Sidama Region": "SNNP Region",  # Sidama was created from SNNP after 2016
}
dhis_unique["dhis_region_mapped"] = dhis_unique["orgunitlevel2"].replace(region_map)
dhis_unique["dhis_region_norm"] = dhis_unique["dhis_region_mapped"].apply(norm_region)
dhis_unique = dhis_unique.rename(
    columns={"orgunitlevel2": "dhis_region", "zone": "dhis_zone"}
)

# Load MFR zones
# 1. From regional sheets
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

    # Specific fix for Dire Dawa (DD)
    # DHIS2 uses "Dire Dawa City Administration" as zone. MFR breaks it down into operations.
    # We aggregate MFR DD into one zone to match DHIS2.
    if sheet == "DD":
        df["zone"] = "Dire Dawa City Administration"
    mfr_list.append(df)

# 2. Tigray is only in the main list
main_sheet = pd.read_excel(
    "data/mfr/MFR List All.xlsx", sheet_name="MFR List All - Feb 23"
)
tigray = main_sheet[
    main_sheet["region"].astype(str).str.contains("Tigray", case=False, na=False)
].copy()
mfr_list.append(tigray)

mfr = pd.concat(mfr_list, ignore_index=True)

# MFR SNNP sometimes has "Zone" in separate column or part of name
mfr_zones = mfr[["region", "zone"]].drop_duplicates().dropna(subset=["zone"])
mfr_zones = mfr_zones[mfr_zones["zone"].str.strip() != ""]
mfr_zones["mfr_zone_norm"] = mfr_zones["zone"].apply(norm)
mfr_zones["mfr_region_norm"] = mfr_zones["region"].apply(norm_region)
mfr_zones = mfr_zones.rename(columns={"region": "mfr_region", "zone": "mfr_zone"})

# Load GeoJSON zones
with open("data/admin_bounds/eth_admin_boundaries.geojson/eth_admin2.geojson") as f:
    geo = json.load(f)
geo_zones = pd.DataFrame(
    [
        {
            "geo_zone": f["properties"]["adm2_name"],
            "geo_pcode": f["properties"]["adm2_pcode"],
            "geo_region": f["properties"]["adm1_name"],
        }
        for f in geo["features"]
    ]
)
geo_zones["geo_zone_norm"] = geo_zones["geo_zone"].apply(norm)

# Manual Map for DHIS2 Zones that don't match MFR directly (Towns to Parent Zones, Split Zones, Spelling)
# Keys are NORMALIZED DHIS2 names. Values are NORMALIZED MFR names.
MANUAL_MAPPING = {
    "debrebirhan": "north shewa",
    "debremarkos": "east gojam",
    "debretabor": "south gondar",
    "kombolcha": "south wollo",
    "woldia": "north wollo",
    "kebena": "gurage",
    "mareko": "gurage",
    "east gurage": "gurage",
    "tembaro": "kembata tembaro",
    "kembata": "kembata tembaro",
    "sheka": "sheko",
}

# Apply manual mapping
dhis_unique["dhis_zone_norm"] = dhis_unique["dhis_zone_norm"].replace(MANUAL_MAPPING)

# Create a specific norm for GeoJSON matching to handle consolidated regions/zones
dhis_unique["dhis_geo_norm"] = dhis_unique["dhis_zone_norm"]

# GeoJSON Specific Mappings
GEO_SPECIFIC_MAPPING = {
    "agniwa": "agnewak",
    "ale": "alle",
    "dawa": "daawa",
    "dawro": "dawuro",
    "kaffa": "kefa",
    "korahay": "korahe",
    "shebelle": "shabelle",
    "dollo": "doolo",
    "koore": "amaro",
    "liben": "liban",
    "nuer": "nuwer",
    "sitti": "siti",
    "west omo": "mirab omo",
    "sheko": "sheka",  # MFR uses Sheko, GeoJSON uses Sheka
    "gurage": "guraghe",
    "silte": "siltie",
    "erar": "erer",
    "gardula": "derashe",
    "aari": "south omo",  # Aari is seemingly the same zone: https://en.wikipedia.org/wiki/South_Omo_Zone
    "north gojam": "west gojam",  # It looks like that on the map, not really sure here?
    "dire dawa city administration": "dire dawa urban",
    # Amhara / Wello / Wag Hamra
    "north wollo": "north wello",
    "south wollo": "south wello",
    "waghimera": "wag hamra",
    # Amhara Towns
    "bahirdar": "west gojam",  # https://en.wikipedia.org/wiki/West_Gojjam_Zone
    "gondar": "central gondar",
    "dessie": "south wello",  # https://en.wikipedia.org/wiki/South_Wollo_Zone
    "kombolcha": "south wello",
    "woldia": "north wello",  # Maybe this is "Wogel Tena" on Wikipedia?
    # Addis Ababa
    "addis ketema": "region 14",
    "akaki kality": "region 14",
    "arada": "region 14",
    "bole": "region 14",
    "gulele": "region 14",
    "kirkos": "region 14",
    "kolfe": "region 14",
    "lemi kura": "region 14",
    "lideta": "region 14",
    "nifas silk lafto": "region 14",
    "yeka": "region 14",
    # Harari
    "abadir": "harari",
    "aboker": "harari",
    "amir nur": "harari",
    "dire teyara": "harari",
    "erer": "harari",
    "hakim": "harari",
    "jinela": "harari",
    "shenkor": "harari",
    "sofi": "harari",
    # Sidama
    "central sidama": "sidama",
    "eastern sidama": "sidama",
    "northern sidama": "sidama",
    "southern sidama": "sidama",
    "hawassa": "sidama",
    "hawassa city": "sidama",
}
dhis_unique["dhis_geo_norm"] = dhis_unique["dhis_geo_norm"].replace(
    GEO_SPECIFIC_MAPPING
)


# Region-Aware Mappings for GeoJSON (North Shewa)
def apply_region_geo_norm(row):
    z = row["dhis_geo_norm"]
    r = row["dhis_region"]
    if z == "north shewa":
        if "Amhara" in str(r):
            return "north shewa am"
        elif "Oromia" in str(r):
            return "north shewa or"
    return z


dhis_unique["dhis_geo_norm"] = dhis_unique.apply(apply_region_geo_norm, axis=1)

# Three-way matching
# Match DHIS2 with MFR
dhis_mfr = dhis_unique.merge(
    mfr_zones,
    left_on=["dhis_zone_norm", "dhis_region_norm"],
    right_on=["mfr_zone_norm", "mfr_region_norm"],
    how="left",
)

# Match with GeoJSON using dhis_geo_norm
mapping = dhis_mfr.merge(
    geo_zones, left_on="dhis_geo_norm", right_on="geo_zone_norm", how="left"
)

# ---------------------------------------------------------------------
# Load Shapefile zones (Admin 2)
# ---------------------------------------------------------------------
shp_path = "data/geography/Ethiopia_adm2_uscb_2016.shp"
shp_gdf = gpd.read_file(shp_path)


# Extract Region from GEO_MATCH (e.g., "ETHIOPIA_AMARA_NORTH SHEWA")
# Structure seems to be COUNTRY_REGION_ZONE
def extract_shp_region(geo_match):
    if not geo_match:
        return ""
    parts = geo_match.split("_")
    if len(parts) >= 2:
        return parts[1]  # e.g. AMARA, OROMIYA
    return ""


shp_zones = pd.DataFrame(
    {
        "shp_zone": shp_gdf["AREA_NAME"],
        "shp_geo_match": shp_gdf["GEO_MATCH"],
        "shp_raw_region": shp_gdf["GEO_MATCH"].apply(extract_shp_region),
    }
)

# Normalize Shapefile Zone Names
shp_zones["shp_zone_norm"] = shp_zones["shp_zone"].apply(norm)

# Manual Map for Shapefile Regions to match DHIS2/MFR Norm
# Based on GEO_MATCH: {'AMARA', 'OROMIYA', 'AFAR', 'BINSHANGUL GUMUZ', 'DIRE DAWA',
# 'GAMBELA HIZBOCH', 'HARERI HIZB', 'YEDEBUB BIHEROCH BIHERESEBOCH NA HIZBOCH', 'SUMALE', 'TIGRAY', 'ADIS ABEBA'}
SHP_REGION_MAP = {
    "AMARA": "Amhara",
    "OROMIYA": "Oromia",
    "AFAR": "Afar",
    "BINSHANGUL GUMUZ": "Benishangul Gumuz",
    "DIRE DAWA": "Dire Dawa",
    "GAMBELA HIZBOCH": "Gambella",
    "HARERI HIZB": "Harari",
    "YEDEBUB BIHEROCH BIHERESEBOCH NA HIZBOCH": "SNNP Region",
    "SUMALE": "Somali",
    "TIGRAY": "Tigray",
    "ADIS ABEBA": "Addis Ababa",
    "REGION 17": "Oromia",  # Fallback for special enumeration areas mostly in Oromia
}
shp_zones["shp_region_mapped"] = shp_zones["shp_raw_region"].replace(SHP_REGION_MAP)
shp_zones["shp_region_norm"] = shp_zones["shp_region_mapped"].apply(norm_region)

# DHIS2 Norm for Shapefile
# Start with standard norm (using 'mapping' df which has these cols)
mapping["dhis_shp_norm"] = mapping["dhis_zone_norm"]

# Shapefile Specific Name Mappings
SHP_SPECIFIC_MAPPING = {
    "agniwa": "agnewak",
    "gurage": "guraghe",
    "silte": "siltie",
    "korahe": "korahay",
    "kefa": "kaffa",
    "dawuro": "dawro",
}
# Add important ones from data inspection
SHP_SPECIFIC_MAPPING.update(
    {
        "nuer": "nuwer",
        "wag hamra": "wag himra",
        "waghimera": "wag himra",
        "west omo": "mirab omo",
        "majang": "mezhenger",
        "halaba": "alaba",
        # Addis Ababa
        "akaki kality": "akaki kaliti",
        "nifas silk lafto": "nefas silklafto",
        "nefas silk lafto": "nefas silklafto",
        "kolfe": "kolfe keraniyo",
        "lemi kura": "bole",
        # Tigray
        "western": "western tigray",
        "eastern": "eastern tigray",
        "central": "central tigray",
        "southern": "southern tigray",
        "north western": "north western tigray",
        "mekelle": "mekele",
        "south eastern": "southern tigray",
        # Harari - All woredas map to the single Harari zone in shapefile
        "abadir": "hareri hizb",
        "aboker": "hareri hizb",
        "dire teyara": "hareri hizb",
        "jinela": "hareri hizb",
        "hakim": "hareri hizb",
        "amir nur": "hareri hizb",
        "erer": "hareri hizb",
        "sofi": "hareri hizb",
        "shenkor": "hareri hizb",
        # Sidama - New region created from SNNP, doesn't exist in 2016 shapefile
        # Map to parent SNNP region would require SNNP zone lookup which doesn't exist cleanly
        # Leave unmatched as shapefile doesn't have Sidama region
        "central sidama": "sidama",
        "eastern sidama": "sidama",
        "northern sidama": "sidama",
        "southern sidama": "sidama",
        "hawassa": "sidama",
        "hawassa city": "sidama",
        # Somali - Zones renamed after 2016
        "jarar": "degehabur",
        "dollo": "warder",
        "shabelle": "gode",
        "shebelle": "gode",
        "nogob": "fik",
        "sitti": "shinile",
        "daawa": "liben",
        "dawa": "liben",
        "fafan": "jijiga",
        "korahay": "korahe",
        "erar": "shinile",  # Erar is part of Sitti/Shinile zone
        # Amhara
        "oromia": "oromiya",  # This maps to ETHIOPIA_AMARA_OROMIYA
        "awi": "awizone",
        "central gondar": "north gondar",
        "west gondar": "north gondar",
        "gondar town": "north gondar",
        "gondar": "north gondar",
        "north wollo": "north wello",
        "south wollo": "south wello",
        "kombolcha": "south wello",
        "dessie": "south wello",
        "woldia": "north wello",
        "north gojjam": "west gojam",
        "north gojam": "west gojam",
        # Benishangul Gumuz
        "assosa": "asossa",
        "kamashi": "kemashi",
        "mao komo": "metekel",  # Mao Komo is part of Metekel zone
        # Dire Dawa
        "dire dawa city": "dir dawa",
        # Gambella - Towns/Special Woredas don't have direct zone equivalents
        # The shapefile has 3 zones: Agnewak, Mezhenger, Nuwer
        # Gambella Town and Itang are administrative but not zone-level in shapefile
        # Leave unmatched as they're not in the shapefile structure
        # Afar - Zone numbering issue
        # Zone 6 doesn't exist in 2016 shapefile (only has 1-5)
        # This is likely a new administrative division
        # Leave unmatched
        # Oromia
        "jimma town": "jimma",
        "adama town": "adama",
        "bishoftu town": "east shewa",
        "shashemene town": "west arsi",
        "west guji": "guji",
        "east guji": "guji",
        "bule hora": "guji",
        # SNNP / Central / South / South West regions
        # These modern regions (Central Ethiopian, South Ethiopia, South West Ethiopia)
        # were created by splitting SNNP after 2016
        # The 2016 shapefile has old SNNP zones that no longer cleanly map
        # We need to map specific zones to their SNNP equivalents
        # Central Ethiopian region zones -> Old SNNP zones
        "east gurage": "gurage",  # shapefile: GURAGE
        "gurage": "gurage",
        "kebena": "gurage",
        "kembata": "kembata timbaro",  # shapefile: KEMBATA TIMBARO
        "mareko": "gurage",
        "silte": "silti",  # shapefile: SILTI
        "siltie": "silti",
        "tembaro": "kembata timbaro",  # shapefile: KEMBATA TIMBARO
        # South West Ethiopia zones -> Old SNNP zones
        "bench sheko": "bench maji",
        "dawro": "dawuro",
        "kaffa": "keffa",  # shapefile: KEFFA
        "kefa": "keffa",
        "konta": "keffa",
        "sheka": "sheka",  # shapefile: SHEKA
        "sheko": "sheka",
        # South Ethiopia zones -> Old SNNP zones
        "aari": "south omo",
        "ale": "south omo",  # Ale was part of South Omo
        "alle": "south omo",
        "basketo": "gamo gofa",  # Basketo was part of Gamo Gofa
        "gamo": "gamo gofa",  # shapefile: GAMO GOFA
        "gofa": "gamo gofa",  # shapefile: GAMO GOFA
        "burji": "gedeo",
        "gardula": "gedeo",
        "dirashe": "gedeo",
        "derashe": "gedeo",
        "konso": "gedeo",
        "koore": "gedeo",  # Koore/Amaro was part of Gedeo area
    }
)

mapping["dhis_shp_norm"] = mapping["dhis_shp_norm"].replace(SHP_SPECIFIC_MAPPING)

# Merge DHIS with Shapefile
# Using Region-Aware Merge
mapping = mapping.merge(
    shp_zones,
    left_on=["dhis_shp_norm", "dhis_region_norm"],
    right_on=["shp_zone_norm", "shp_region_norm"],
    how="left",
)


# Format output

# Format output
mapping = mapping.rename(
    columns={
        "dhis_region": "region",
        "dhis_zone": "zone_dhis2",
        "mfr_zone": "zone_mfr",
        "geo_zone": "zone_geojson",
        "shp_zone": "zone_shapefile",
        "shp_geo_match": "zone_shapefile_geouid",
    }
)
output_cols = [
    "region",
    "zone_dhis2",
    "zone_mfr",
    "zone_geojson",
    "zone_shapefile",
    "zone_shapefile_geouid",
]
mapping = mapping[output_cols].drop_duplicates()
mapping = mapping[mapping["zone_dhis2"].notna()]

# Save mapping
mapping.to_csv("estimates/zone_three_way_mapping.csv", index=False)

# Calculate match statistics
total_dhis = mapping["zone_dhis2"].nunique()
dhis_matched_mfr = mapping[mapping["zone_mfr"].notna()]["zone_dhis2"].nunique()
dhis_matched_geo = mapping[mapping["zone_geojson"].notna()]["zone_dhis2"].nunique()
dhis_matched_shp = mapping[mapping["zone_shapefile"].notna()]["zone_dhis2"].nunique()
all_three_count = mapping[
    (mapping["zone_mfr"].notna()) & (mapping["zone_geojson"].notna())
]["zone_dhis2"].nunique()

print(f"\nSaved {len(mapping)} total mappings to estimates/zone_three_way_mapping.csv")
print("\nZones by source:")
print(f"  DHIS2 Unique Zones: {total_dhis}")

print("\nMatch percentages (DHIS2 based):")
print(
    f"  DHIS2 → GeoJSON: {dhis_matched_geo}/{total_dhis} ({100*dhis_matched_geo/total_dhis:.1f}%)"
)
print(
    f"  DHIS2 → Shapefile: {dhis_matched_shp}/{total_dhis} ({100*dhis_matched_shp/total_dhis:.1f}%)"
)
print(
    f"  DHIS2 → MFR:     {dhis_matched_mfr}/{total_dhis} ({100*dhis_matched_mfr/total_dhis:.1f}%)"
)

# MFR -> GeoJSON Match Rate
total_mfr = mfr_zones["mfr_zone"].nunique()
mfr_matched_geo = mfr_zones[
    mfr_zones["mfr_zone_norm"].isin(geo_zones["geo_zone_norm"])
]["mfr_zone"].nunique()
print(
    f"  MFR → GeoJSON:   {mfr_matched_geo}/{total_mfr} ({100*mfr_matched_geo/total_mfr:.1f}%)"
)


print(f"  All three:       {all_three_count} ({100*all_three_count/total_dhis:.1f}%)")
