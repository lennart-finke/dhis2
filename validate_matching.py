"""
Validate GPS vs name-based boundary matching agreement.

This script compares two methods for assigning DHIS2 facilities to administrative units:
  - GPS-based: Uses facility GPS coordinates to spatially match to GeoJSON boundaries
  - Name-based: Uses DHIS2 organisation unit hierarchy names to match to GeoJSON

The analysis pipeline now defaults to the name-based method for consistency.
This script helps validate how well the two methods agree, useful for understanding
potential sources of error in administrative unit assignment.
"""

import pandas as pd
import json

from match_woreda_names import norm, norm_region

print("Validating GPS vs name-based boundary matching agreement")
print("=" * 60)
print()

# Load facility boundary mapping (GPS-based, with name fallback)
facility_map = pd.read_csv("estimates/facility_boundary_mapping.csv")

# Load woreda name mapping (from match_woreda_names.py)
woreda_map = pd.read_csv("estimates/woreda_three_way_mapping.csv")

# Load GeoJSON to get zone/region for each woreda (for name-based lookup)
with open("data/admin_bounds/eth_admin_boundaries.geojson/eth_admin3.geojson") as f:
    geo = json.load(f)
geo_lookup = pd.DataFrame(
    [
        {
            "geo_woreda": feat["properties"]["adm3_name"],
            "geo_zone": feat["properties"]["adm2_name"],
            "geo_region": feat["properties"]["adm1_name"],
        }
        for feat in geo["features"]
    ]
)

# Focus on GPS-matched facilities for comparison
gps_matched = facility_map[facility_map["match_method"] == "gps"].copy()
print(f"Facilities matched by GPS: {len(gps_matched)}")

# Urban regions use dhis2_zone as woreda (DHIS2 hierarchy is flat for these)
urban_regions = {"Addis Ababa", "Harari", "Dire Dawa"}
# Map DHIS2 region names to boundary_region names for matching
DHIS2_TO_BOUNDARY_REGION = {
    "Addis Ababa City Administration": "Addis Ababa",
    "Harari Region": "Harari",
    "Dire Dawa City Administration": "Dire Dawa",
}
gps_matched["boundary_region_check"] = gps_matched["boundary_region"]
is_urban = gps_matched["dhis2_region"].map(DHIS2_TO_BOUNDARY_REGION).isin(urban_regions)

# For urban regions, use dhis2_zone as the woreda lookup key (it contains actual
# woreda/sub-city names like "Abadir Woreda", "Bole Sub City")
# For non-urban, use dhis2_woreda as before
gps_matched["lookup_key"] = gps_matched["dhis2_woreda"]
gps_matched.loc[is_urban, "lookup_key"] = gps_matched.loc[is_urban, "dhis2_zone"]

gps_matched["lookup_key_norm"] = gps_matched["lookup_key"].apply(norm)

# Get name-based boundary assignment (woreda_geojson from mapping)
woreda_map["woreda_dhis2_norm"] = woreda_map["woreda_dhis2"].apply(norm)
woreda_map["zone_norm"] = woreda_map["zone"].apply(norm)

woreda_map["region_norm"] = woreda_map["region"].apply(norm_region)

# Build name lookup with zone and region info
name_lookup = woreda_map[
    ["woreda_dhis2_norm", "zone_norm", "region_norm", "woreda_geojson"]
].drop_duplicates()
name_lookup = name_lookup.rename(columns={"woreda_geojson": "name_woreda"})

# Add zone/region from GeoJSON data (so we compare like-to-like)
name_lookup = name_lookup.merge(
    geo_lookup, left_on="name_woreda", right_on="geo_woreda", how="left"
)
name_lookup = name_lookup.rename(
    columns={"geo_zone": "name_zone", "geo_region": "name_region"}
)

# Also normalize dhis2_zone and dhis2_region for disambiguation
gps_matched["dhis2_zone_norm"] = gps_matched["dhis2_zone"].apply(norm)
gps_matched["dhis2_region_norm"] = gps_matched["dhis2_region"].apply(norm_region)

# Merge: join on normalized woreda name, then prefer rows where region+zone match
comparison = gps_matched.merge(
    name_lookup, left_on="lookup_key_norm", right_on="woreda_dhis2_norm", how="left"
)

# Disambiguation: prefer region match first, then zone match
comparison["_region_match"] = (
    comparison["dhis2_region_norm"] == comparison["region_norm"]
)
comparison["_zone_match"] = comparison["dhis2_zone_norm"] == comparison["zone_norm"]
comparison = comparison.sort_values(
    ["_region_match", "_zone_match"], ascending=[False, False]
)
# Use the stable facility identifier as the dedup key, not all gps_matched columns
# (which would silently grow if new columns are added).
comparison = comparison.drop_duplicates(subset=["dhis2_id"], keep="first")
comparison = comparison.drop(columns=["_region_match", "_zone_match"])

# After disambiguation there must be exactly one row per GPS-matched facility.
# Any violation means two name candidates tied on region+zone score — a genuine ambiguity.
n_facilities = len(gps_matched["dhis2_id"].unique())
n_after_dedup = len(comparison["dhis2_id"].unique())
assert n_after_dedup == n_facilities, (
    f"Deduplication did not collapse to one row per facility: "
    f"{n_after_dedup} unique dhis2_ids after dedup vs {n_facilities} GPS-matched facilities"
)

# Normalize for comparison
comparison["gps_woreda_norm"] = comparison["boundary_woreda"].apply(norm)
comparison["gps_zone_norm"] = comparison["boundary_zone"].apply(norm)
comparison["gps_region_norm"] = comparison["boundary_region"].apply(norm)
comparison["name_woreda_norm"] = comparison["name_woreda"].apply(norm)
comparison["name_zone_norm"] = comparison["name_zone"].apply(norm)
comparison["name_region_norm"] = comparison["name_region"].apply(norm)

# Only compare where both methods assigned something
has_both = comparison["name_woreda"].notna()
comp = comparison[has_both]
print(f"Facilities with both GPS and name-based assignment: {len(comp)}")

# Compute agreement rates
comp = comp.copy()  # Avoid SettingWithCopyWarning
comp["is_region_match"] = comp["gps_region_norm"] == comp["name_region_norm"]
comp["is_zone_match"] = comp["gps_zone_norm"] == comp["name_zone_norm"]
comp["is_woreda_match"] = comp["gps_woreda_norm"] == comp["name_woreda_norm"]

woreda_match = comp["is_woreda_match"].sum()
zone_match = comp["is_zone_match"].sum()
region_match = comp["is_region_match"].sum()

print("\nAgreement rates (GPS vs name-based matching):")
print(f"  Region: {region_match}/{len(comp)} ({100*region_match/len(comp):.1f}%)")
print(f"  Zone:   {zone_match}/{len(comp)} ({100*zone_match/len(comp):.1f}%)")
print(f"  Woreda: {woreda_match}/{len(comp)} ({100*woreda_match/len(comp):.1f}%)")

print("\nAgreement rates by Region (sorted by Woreda match %):")
stats = comp.groupby("boundary_region")[
    ["is_region_match", "is_zone_match", "is_woreda_match"]
].agg(["sum", "count", "mean"])

# Flatten columns for easier access
stats.columns = ["_".join(col).strip() for col in stats.columns.values]
stats = stats.sort_values("is_woreda_match_mean", ascending=False)

for region, row in stats.iterrows():
    count = int(row["is_woreda_match_count"])
    w_match = row["is_woreda_match_mean"] * 100
    z_match = row["is_zone_match_mean"] * 100
    r_match = row["is_region_match_mean"] * 100
    print(
        f"  {region:<20} (n={count:<3}): Region={r_match:5.1f}% | Zone={z_match:5.1f}% | Woreda={w_match:5.1f}%"
    )

# Show disagreements
disagreements = comp[comp["gps_woreda_norm"] != comp["name_woreda_norm"]]
if len(disagreements) > 0:
    print(f"\nWoreda disagreements ({len(disagreements)}):")
    for _, row in disagreements.head(10).iterrows():
        print(
            f"  {row['facility_name'][:40]}: GPS={row['boundary_woreda']} vs Name={row['name_woreda']}"
        )
