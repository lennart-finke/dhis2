"""Match DHIS2 facilities to admin boundaries using GPS or name matching.

Reads the output of match_facility_names.py and:
1. For facilities with valid GPS coordinates from MFR, performs spatial join
   with admin boundaries to get region, zone, woreda from shapefile
2. For facilities without GPS, falls back to DHIS2 name-matched boundaries,
   then maps zone/woreda names to GeoJSON names using the output of
   match_zone_names.py and match_woreda_names.py to ensure consistency
   with survey data

Outputs: estimates/facility_boundary_mapping.csv
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load facility mapping from match_facility_names.py
facility_mapping = pd.read_csv("estimates/facility_mapping.csv")

# Load all facilities GPS file
facilities_gps = pd.read_csv("data/dhis2/gps/facilities.csv")
# Keep only facilities with valid GPS coordinates
facilities_gps = facilities_gps[
    facilities_gps["latitude"].notna()
    & facilities_gps["longitude"].notna()
    & (facilities_gps["latitude"] != 0)
    & (facilities_gps["longitude"] != 0)
][["facility_id", "latitude", "longitude"]].copy()

facilities_gps = facilities_gps.rename(
    columns={
        "facility_id": "facility_mfr_id",
        "latitude": "lat_new",
        "longitude": "lon_new",
    }
)

# Merge GPS with facility mapping
facility_mapping = facility_mapping.merge(
    facilities_gps, on="facility_mfr_id", how="left"
)

# Override lat/lon where available
gps_mask = facility_mapping["lat_new"].notna()
facility_mapping.loc[gps_mask, "lat"] = facility_mapping.loc[gps_mask, "lat_new"]
facility_mapping.loc[gps_mask, "lon"] = facility_mapping.loc[gps_mask, "lon_new"]
# Drop the temporary columns
facility_mapping = facility_mapping.drop(columns=["lat_new", "lon_new"])

print(
    f"Loaded {len(facilities_gps)} facilities with GPS coordinates from facilities.csv"
)
print(f"Merged {gps_mask.sum()} GPS coordinates into facility mapping")

# Fallback 1: Afar_facilities.csv (same schema, covers facilities not in facilities.csv)
afar_gps = pd.read_csv("data/dhis2/gps/Afar_facilities.csv")
afar_gps = afar_gps[
    afar_gps["latitude"].notna()
    & afar_gps["longitude"].notna()
    & (afar_gps["latitude"] != 0)
    & (afar_gps["longitude"] != 0)
][["facility_id", "latitude", "longitude"]].copy()
afar_gps = afar_gps.rename(
    columns={
        "facility_id": "facility_mfr_id",
        "latitude": "lat_afar",
        "longitude": "lon_afar",
    }
)

facility_mapping = facility_mapping.merge(afar_gps, on="facility_mfr_id", how="left")
# Only fill where lat/lon still missing
still_missing = facility_mapping["lat"].isna() | (facility_mapping["lat"] == 0)
afar_fill = still_missing & facility_mapping["lat_afar"].notna()
facility_mapping.loc[afar_fill, "lat"] = facility_mapping.loc[afar_fill, "lat_afar"]
facility_mapping.loc[afar_fill, "lon"] = facility_mapping.loc[afar_fill, "lon_afar"]
facility_mapping = facility_mapping.drop(columns=["lat_afar", "lon_afar"])
print(
    f"Fallback 1 (Afar_facilities.csv): filled {afar_fill.sum()} additional facilities"
)

# Fallback 2: Facilities.geojson (DHIS2 facility points, merge on DHIS2 ID)
geo_facilities = gpd.read_file("data/dhis2/gps/Facilities.geojson")
geo_facilities = geo_facilities[geo_facilities.geometry.notna()].copy()
geo_facilities["lat_geo"] = geo_facilities.geometry.y
geo_facilities["lon_geo"] = geo_facilities.geometry.x
geo_facilities = geo_facilities[
    (geo_facilities["lat_geo"] != 0) & (geo_facilities["lon_geo"] != 0)
][["id", "lat_geo", "lon_geo"]].copy()
geo_facilities = geo_facilities.rename(columns={"id": "facility_dhis2_id"})
# Convert to regular DataFrame for merge
geo_facilities = pd.DataFrame(geo_facilities)

facility_mapping["facility_dhis2_id"] = facility_mapping["facility_dhis2_id"].astype(
    str
)
geo_facilities["facility_dhis2_id"] = geo_facilities["facility_dhis2_id"].astype(str)
facility_mapping = facility_mapping.merge(
    geo_facilities, on="facility_dhis2_id", how="left"
)
# Only fill where lat/lon still missing
still_missing = facility_mapping["lat"].isna() | (facility_mapping["lat"] == 0)
geo_fill = still_missing & facility_mapping["lat_geo"].notna()
facility_mapping.loc[geo_fill, "lat"] = facility_mapping.loc[geo_fill, "lat_geo"]
facility_mapping.loc[geo_fill, "lon"] = facility_mapping.loc[geo_fill, "lon_geo"]
facility_mapping = facility_mapping.drop(columns=["lat_geo", "lon_geo"])
print(
    f"Fallback 2 (Facilities.geojson): filled {geo_fill.sum()} additional facilities\n"
)


# Load admin boundaries (woreda level = admin3)
admin3 = gpd.read_file(
    "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin3.geojson"
)

# Ethiopia bounding box (approximate)
# Latitude: 3°N to 15°N
# Longitude: 33°E to 48°E
ETH_LAT_MIN, ETH_LAT_MAX = 3.0, 15.0
ETH_LON_MIN, ETH_LON_MAX = 33.0, 48.0

# Identify facilities with valid GPS coordinates
# Must be non-null, non-zero, AND within Ethiopia's bounding box
has_coords = (
    facility_mapping["lat"].notna()
    & facility_mapping["lon"].notna()
    & (facility_mapping["lat"] != 0)
    & (facility_mapping["lon"] != 0)
)
coords_in_ethiopia = (
    (facility_mapping["lat"] >= ETH_LAT_MIN)
    & (facility_mapping["lat"] <= ETH_LAT_MAX)
    & (facility_mapping["lon"] >= ETH_LON_MIN)
    & (facility_mapping["lon"] <= ETH_LON_MAX)
)

# Check for swapped coordinates (lat in lon range, lon in lat range)
swapped_in_ethiopia = (
    (facility_mapping["lon"] >= ETH_LAT_MIN)
    & (facility_mapping["lon"] <= ETH_LAT_MAX)
    & (facility_mapping["lat"] >= ETH_LON_MIN)
    & (facility_mapping["lat"] <= ETH_LON_MAX)
)

# Identify valid sets
valid_normal = has_coords & coords_in_ethiopia
valid_swapped = has_coords & ~coords_in_ethiopia & swapped_in_ethiopia

# Apply fixes
facility_mapping.loc[valid_swapped, ["lat", "lon"]] = facility_mapping.loc[
    valid_swapped, ["lon", "lat"]
].values

# Re-evaluate valid GPS after swap match
has_valid_gps = valid_normal | valid_swapped

# Count invalid coords for reporting
invalid_coords = has_coords & ~has_valid_gps
n_invalid = invalid_coords.sum()
n_swapped = valid_swapped.sum()

facilities_with_gps = facility_mapping[has_valid_gps].copy()
facilities_without_gps = facility_mapping[~has_valid_gps].copy()

# Map from DHIS2 region names to normalized GeoJSON region names
DHIS2_TO_NORMALIZED_REGION = {
    "Addis Ababa City Administration": "Addis Ababa",
    "Addis Ababa Region": "Addis Ababa",
    "Dire Dawa City Administration": "Dire Dawa",
    "Dire Dawa Region": "Dire Dawa",
    "Tigray Region": "Tigray",
    "Amhara Region": "Amhara",
    "Oromia Region": "Oromia",
    "Somali Region": "Somali",
    "Afar Region": "Afar",
    "Harari Region": "Harari",
    "Gambella Region": "Gambela",
    "Gambela Region": "Gambela",
    "Benishangul Gumz Region": "Benishangul Gumz",
    "Benishangul-Gumuz Region": "Benishangul Gumz",
    "Benishangul Gumuz Regional Health Bureau": "Benishangul Gumz",
    "SNNPR": "SNNP",
    "SNNP Region": "SNNP",
    "South West Ethiopia Region": "South West Ethiopia",
    "Sidama Region": "Sidama",
    "Central Ethiopian region": "SNNP",
    "South Ethiopia": "SNNP",
    "South Ethiopia Regional State": "SNNP",
    "South Ethiopia Region": "SNNP",
}

print(f"Total facilities: {len(facility_mapping)}")
print(f"  With valid GPS (in Ethiopia): {len(facilities_with_gps)}")
print(f"    (includes {n_swapped} facilities with swapped lat/lon that were fixed)")
print(f"  Without GPS or invalid coords: {len(facilities_without_gps)}")
print(f"    (includes {n_invalid} facilities with coords outside Ethiopia)")

# Create GeoDataFrame for facilities with GPS and perform spatial join
assert len(facilities_with_gps) > 0
geometry = [
    Point(xy) for xy in zip(facilities_with_gps["lon"], facilities_with_gps["lat"])
]
facilities_gdf = gpd.GeoDataFrame(
    facilities_with_gps, geometry=geometry, crs="EPSG:4326"
)

# Spatial join to get admin boundaries (Woreda Level - Admin 3)
facilities_joined_admin3 = gpd.sjoin(
    facilities_gdf,
    admin3[["adm1_name", "adm2_name", "adm3_name", "adm3_pcode", "geometry"]],
    how="left",
    predicate="within",
)
# Handle overlapping boundaries by keeping the first match
facilities_joined_admin3 = facilities_joined_admin3[
    ~facilities_joined_admin3.index.duplicated(keep="first")
]

# Rename boundary columns from Admin 3
# Note: We initially take region/zone/woreda from Admin 3, but we'll override Zone with Admin 2 later
facilities_joined_admin3 = facilities_joined_admin3.rename(
    columns={
        "adm1_name": "boundary_region",
        "adm2_name": "boundary_zone_raw",  # Potentially incorrect zone (e.g. Region 14)
        "adm3_name": "boundary_woreda",
        "adm3_pcode": "boundary_woreda_pcode",
    }
)

# Drop index_right from first join to allow second join
facilities_joined_admin3 = facilities_joined_admin3.drop(
    columns=["index_right"], errors="ignore"
)

# Spatial join to get admin boundaries (Zone Level - Admin 2)
# This provides the correct zone names (e.g. Addis subcities instead of "Region 14")
# Load admin2 if not already loaded
admin2 = gpd.read_file(
    "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin2.geojson"
)

facilities_joined_admin2 = gpd.sjoin(
    facilities_gdf,
    admin2[["adm2_name", "geometry"]],
    how="left",
    predicate="within",
)
facilities_joined_admin2 = facilities_joined_admin2[
    ~facilities_joined_admin2.index.duplicated(keep="first")
]

# We only need the valid zone name from Admin 2
facilities_joined_admin2 = facilities_joined_admin2[["adm2_name"]]
facilities_joined_admin2 = facilities_joined_admin2.rename(
    columns={"adm2_name": "boundary_zone"}
)

# Spatial join to get admin boundaries (Kebele Level - Admin 4)
admin4 = gpd.read_file(
    "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin4.geojson"
)
facilities_joined_admin4 = gpd.sjoin(
    facilities_gdf,
    admin4[["R_NAME", "Z_NAME", "W_NAME", "RK_NAME", "geometry"]],
    how="left",
    predicate="within",
)
facilities_joined_admin4 = facilities_joined_admin4[
    ~facilities_joined_admin4.index.duplicated(keep="first")
]
facilities_joined_admin4 = facilities_joined_admin4[
    ["R_NAME", "Z_NAME", "W_NAME", "RK_NAME"]
].rename(
    columns={
        "R_NAME": "boundary_region_admin4",
        "Z_NAME": "boundary_zone_admin4",
        "W_NAME": "boundary_woreda_admin4",
        "RK_NAME": "boundary_kebele",
    }
)

# Combine the results (both have the same index as facilities_gdf/facilities_with_gps)
facilities_joined = pd.concat(
    [
        facilities_joined_admin3.drop(columns=["boundary_zone_raw"], errors="ignore"),
        facilities_joined_admin2,
        facilities_joined_admin4,
    ],
    axis=1,
)

# For Addis Ababa and Harari, the population data uses the Zone/Sub-City name as the Woreda.
# However, determining 'woreda' from spatial join (Admin 3) gives specific numbered woredas.
# We must overwrite 'boundary_woreda' with 'boundary_zone' (Admin 2) for these regions
# to match the population data granularity.
subset_mask = facilities_joined["boundary_region"].isin(["Addis Ababa", "Harari"])
facilities_joined.loc[subset_mask, "boundary_woreda"] = facilities_joined.loc[
    subset_mask, "boundary_zone"
]

facilities_joined["match_method"] = "gps"

# Region-consistency check: reject GPS matches where the boundary region
# does not match the facility's DHIS2 region. This prevents cross-region
# mismatches for facilities near regional borders.
facilities_joined["dhis2_region_normalized"] = (
    facilities_joined["region"]
    .map(DHIS2_TO_NORMALIZED_REGION)
    .fillna(facilities_joined["region"])
)
region_mismatch = facilities_joined["boundary_region"].notna() & (
    facilities_joined["dhis2_region_normalized"] != facilities_joined["boundary_region"]
)
n_region_mismatch = region_mismatch.sum()
if n_region_mismatch > 0:
    print(f"\n  Rejected {n_region_mismatch} GPS matches due to region mismatch")
    # Show some examples
    examples = facilities_joined.loc[
        region_mismatch,
        ["facility_dhis2", "region", "dhis2_region_normalized", "boundary_region"],
    ].head(5)
    for _, row in examples.iterrows():
        print(
            f"    {row['facility_dhis2']}: DHIS2={row['dhis2_region_normalized']}, "
            f"GPS boundary={row['boundary_region']}"
        )
    # Clear boundary fields for mismatched facilities
    boundary_cols = [
        "boundary_region",
        "boundary_zone",
        "boundary_woreda",
        "boundary_woreda_pcode",
        "boundary_kebele",
        "boundary_region_admin4",
        "boundary_zone_admin4",
        "boundary_woreda_admin4",
    ]
    for col in boundary_cols:
        if col in facilities_joined.columns:
            facilities_joined.loc[region_mismatch, col] = None
    facilities_joined.loc[region_mismatch, "match_method"] = "gps_region_mismatch"
facilities_joined = facilities_joined.drop(columns=["dhis2_region_normalized"])

# Drop geometry
facilities_joined = pd.DataFrame(
    facilities_joined.drop(columns=["geometry"], errors="ignore")
)

# For facilities without GPS, use DHIS2 name-matched boundaries
# Normalize region names using the dictionary defined earlier
facilities_without_gps["boundary_region"] = (
    facilities_without_gps["region"]
    .map(DHIS2_TO_NORMALIZED_REGION)
    .fillna(facilities_without_gps["region"])
)
facilities_without_gps["boundary_zone"] = facilities_without_gps["zone"]
facilities_without_gps["boundary_woreda"] = facilities_without_gps["woreda"]
facilities_without_gps["boundary_woreda_pcode"] = None
facilities_without_gps["boundary_kebele"] = None
facilities_without_gps["boundary_region_admin4"] = None
facilities_without_gps["boundary_zone_admin4"] = None
facilities_without_gps["boundary_woreda_admin4"] = None
facilities_without_gps["match_method"] = "name"


def clean_zone_name(name):
    if pd.isna(name):
        return None
    name = str(name).strip()

    # Specific mappings
    mappings = {
        "Kolfe": "Kolfe Keraniyo",
        "Kolfe Subcity": "Kolfe Keraniyo",
        "Kolfe Sub City": "Kolfe Keraniyo",
    }
    if name in mappings:
        return mappings[name]

    # Suffix removal (case insensitive)
    suffixes = [
        " Sub City",
        " Sub city",
        " Subcity",
        " SubCity",
        " Woreda",
        " WA",
        " Town",
        " City Administration",
    ]

    name_lower = name.lower()
    for suffix in suffixes:
        if name_lower.endswith(suffix.lower()):
            return name[: len(name) - len(suffix)]

    return name


facilities_without_gps["boundary_zone"] = facilities_without_gps["boundary_zone"].apply(
    clean_zone_name
)

# Load zone and woreda mappings to convert DHIS2 names to GeoJSON names
# This ensures the output uses the same boundary names as the survey data
zone_mapping = pd.read_csv("estimates/zone_three_way_mapping.csv")
woreda_mapping = pd.read_csv("estimates/woreda_three_way_mapping.csv")

# Create lookup dictionaries for DHIS2 -> GeoJSON conversion
zone_lookup = (
    zone_mapping[["zone_dhis2", "zone_geojson"]]
    .dropna()
    .drop_duplicates()
    .set_index("zone_dhis2")["zone_geojson"]
    .to_dict()
)

woreda_lookup = (
    woreda_mapping[["woreda_dhis2", "woreda_geojson", "woreda_geojson_pcode"]]
    .dropna(subset=["woreda_dhis2", "woreda_geojson"])
    .drop_duplicates(subset=["woreda_dhis2"])
)

# Map zone names to GeoJSON for name-matched facilities
facilities_without_gps["boundary_zone"] = (
    facilities_without_gps["boundary_zone"]
    .map(zone_lookup)
    .fillna(facilities_without_gps["boundary_zone"])
)

# Map woreda names to GeoJSON (including pcode)
facilities_without_gps = facilities_without_gps.merge(
    woreda_lookup,
    left_on="boundary_woreda",
    right_on="woreda_dhis2",
    how="left",
)

# Update boundary_woreda and boundary_woreda_pcode with GeoJSON values where available
facilities_without_gps["boundary_woreda"] = facilities_without_gps[
    "woreda_geojson"
].fillna(facilities_without_gps["boundary_woreda"])
facilities_without_gps["boundary_woreda_pcode"] = facilities_without_gps[
    "woreda_geojson_pcode"
].fillna(facilities_without_gps["boundary_woreda_pcode"])

# Drop the temporary merge columns
facilities_without_gps = facilities_without_gps.drop(
    columns=["woreda_dhis2", "woreda_geojson", "woreda_geojson_pcode"], errors="ignore"
)

# For Addis Ababa and Harari, the population data uses the Zone/Sub-City name as the Woreda.
# We must overwrite 'boundary_woreda' with 'boundary_zone' for these regions
# to match the population data granularity, just as we did for GPS-matched facilities.
subset_mask = facilities_without_gps["boundary_region"].isin(["Addis Ababa", "Harari"])
facilities_without_gps.loc[subset_mask, "boundary_woreda"] = facilities_without_gps.loc[
    subset_mask, "boundary_zone"
]

# Combine GPS-matched and name-matched facilities
mapping = pd.concat([facilities_joined, facilities_without_gps], ignore_index=True)

# Rename columns to match expected output format
mapping = mapping.rename(
    columns={
        "facility_dhis2_id": "dhis2_id",
        "facility_dhis2": "facility_name",
        "region": "dhis2_region",
        "zone": "dhis2_zone",
        "woreda": "dhis2_woreda",
        "lat": "latitude",
        "lon": "longitude",
        "facility_mfr_id": "facility_id",
    }
)

# Select output columns
output_cols = [
    "dhis2_id",
    "facility_name",
    "dhis2_region",
    "dhis2_zone",
    "dhis2_woreda",
    "latitude",
    "longitude",
    "facility_id",
    "boundary_kebele",
    "boundary_woreda",
    "boundary_woreda_pcode",
    "boundary_zone",
    "boundary_region",
    "boundary_region_admin4",
    "boundary_zone_admin4",
    "boundary_woreda_admin4",
    "match_method",
]
mapping = mapping[[c for c in output_cols if c in mapping.columns]].drop_duplicates()
mapping = mapping[mapping["facility_name"].notna()]

# Save
mapping.to_csv("estimates/facility_boundary_mapping.csv", index=False)

# Statistics
gps_matched = mapping[mapping["match_method"] == "gps"]["facility_name"].nunique()
name_matched = mapping[mapping["match_method"] == "name"]["facility_name"].nunique()
gps_with_boundary = mapping[
    (mapping["match_method"] == "gps") & mapping["boundary_woreda"].notna()
]["facility_name"].nunique()

print(f"\nSaved {len(mapping)} mappings to estimates/facility_boundary_mapping.csv")
print("\nBoundary assignment statistics:")
print(f"  Assigned via GPS (spatial join):  {gps_matched}")
print(f"    - Successfully matched:         {gps_with_boundary}")
print(f"    - Outside boundaries:           {gps_matched - gps_with_boundary}")
print(f"  Assigned via name matching:       {name_matched}")

# Calculate GPS mapping fraction per region
# Group by DHIS2 region (normalized) so that the denominator and numerator
# refer to the same set of facilities.
print("\nFraction of facilities mapped via GPS per region:")
mapping["dhis2_region_norm"] = (
    mapping["dhis2_region"]
    .map(DHIS2_TO_NORMALIZED_REGION)
    .fillna(mapping["dhis2_region"])
)
region_stats = (
    mapping.groupby("dhis2_region_norm")
    .agg(
        total_facilities=("facility_name", "nunique"),
        gps_mapped=(
            "facility_name",
            lambda x: x[mapping.loc[x.index, "match_method"] == "gps"].nunique(),
        ),
    )
    .reset_index()
    .rename(columns={"dhis2_region_norm": "region"})
)
mapping = mapping.drop(columns=["dhis2_region_norm"])
region_stats["gps_fraction"] = (
    region_stats["gps_mapped"] / region_stats["total_facilities"]
)
region_stats = region_stats.sort_values("gps_fraction", ascending=False)

for _, row in region_stats.iterrows():
    region = row["region"]
    if pd.notna(region):
        print(
            f"  {region:30s}: {row['gps_fraction']:.2%} ({row['gps_mapped']}/{row['total_facilities']})"
        )

national_total = region_stats["total_facilities"].sum()
national_gps = region_stats["gps_mapped"].sum()
assert national_total > 0
national_fraction = national_gps / national_total
print(
    f"  {'National average':30s}: {national_fraction:.2%} ({national_gps}/{national_total})"
)

mfr_matched = facility_mapping[facility_mapping["facility_mfr_id"].notna()][
    "facility_dhis2_id"
].nunique()
total_dhis2 = facility_mapping["facility_dhis2_id"].nunique()
print("\nDHIS2 -> MFR Match Rate:")
print(
    f"  DHIS2 -> MFR:                     {mfr_matched/total_dhis2:.2%} ({mfr_matched}/{total_dhis2})"
)
