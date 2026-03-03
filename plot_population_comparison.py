"""Compare WorldPop and Census 2022 woreda population estimates."""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import numpy as np
from dhis2_utils import get_region_color

worldpop = pd.read_csv("estimates/populations_worldpop_geojson.csv")
census = pd.read_csv("data/census_populations/census_data.csv")
mapping = pd.read_csv("estimates/geojson_census_mapping.csv")

# Filter WorldPop to 2022, woreda level only
wp = worldpop[(worldpop["year"] == 2022) & worldpop["woreda"].notna()].copy()
wp = wp[["region", "woreda", "population"]].rename(
    columns={"population": "pop_worldpop"}
)

# Filter census to woredas
woreda_types = [
    "District",
    "Sub City",
    "Town",
    "Special District",
    "Special Census District",
]
cen = census[census["status"].isin(woreda_types)][
    ["id", "name", "status", "pop_projection_2022"]
].copy()
cen = cen.rename(columns={"id": "census_id", "pop_projection_2022": "pop_census"})

# Create mapping from woreda_geojson -> census_id
# 1. Get unique (woreda_geojson, census_id) pairs
map_subset = mapping[["geo_woreda", "census_id"]].dropna().drop_duplicates()
map_subset = map_subset.rename(
    columns={"geo_woreda": "woreda_geojson", "census_id": "woreda_census_id"}
)
# 2. Join with census status to help disambiguate
map_subset = map_subset.merge(
    cen[["census_id", "status"]],
    left_on="woreda_census_id",
    right_on="census_id",
    how="left",
)


def pick_best_census_match(group):
    """
    If multiple census IDs map to the same GeoJSON woreda name, picking logic:
    - If GeoJSON name contains 'Town' or 'City', prefer Census status 'Town', 'City', or 'Sub City'.
    - Else prefer 'District' or 'Special District'.
    """
    if len(group) == 1:
        return group.iloc[0]["woreda_census_id"]

    woreda_name = group.name.lower()
    is_urban_name = "town" in woreda_name or "city" in woreda_name

    def score(row):
        status = str(row["status"]).lower()
        is_urban_status = "town" in status or "city" in status or "sub city" in status

        # Match urban-ness
        if is_urban_name == is_urban_status:
            return 1
        return 0

    # Apply scoring
    group["score"] = group.apply(score, axis=1)
    # Sort by score descending
    best = group.sort_values("score", ascending=False).iloc[0]
    return best["woreda_census_id"]


woreda_to_census = (
    map_subset.groupby("woreda_geojson").apply(pick_best_census_match).to_dict()
)

# Merge worldpop with census via woreda name -> census_id
wp["census_id"] = wp["woreda"].map(woreda_to_census)

# Aggregate WorldPop estimates by census_id (summing fragments if any)
# Use region from first occurrence
wp_agg = (
    wp.groupby("census_id")
    .agg({"pop_worldpop": "sum", "region": "first"})
    .reset_index()
)

# Merge with Census data
merged = wp_agg.merge(cen, on="census_id", how="inner")

# Normalize region names for color lookup
region_map = {
    "Addis Ababa": "Addis Ababa",
    "Afar": "Afar",
    "Amhara": "Amhara",
    "Benishangul Gumz": "Benishangul Gumuz",
    "Dire Dawa": "Dire Dawa",
    "Gambella": "Gambela",
    "Harari": "Harari",
    "Oromiya": "Oromia",
    "Sidama": "Sidama",
    "Somali": "Somali",
    "SNNP": "SNNP",
    "South West Ethiopia": "South West Ethiopia",
    "Tigray": "Tigray",
}
merged["region_std"] = merged["region"].map(region_map).fillna(merged["region"])

# Statistics
r, _ = pearsonr(merged["pop_census"], merged["pop_worldpop"])
rho, _ = spearmanr(merged["pop_census"], merged["pop_worldpop"])
bias = np.mean(merged["pop_worldpop"] / merged["pop_census"])
print(f"n = {len(merged)}")
print(f"Pearson's r = {r:.3f}")
print(f"Spearman's ρ = {rho:.3f}")
print(f"Bias (mean WorldPop/Census) = {bias:.3f}")

# Plot
fig, ax = plt.subplots(figsize=(6, 6))

for region in merged["region_std"].unique():
    subset = merged[merged["region_std"] == region]
    color = get_region_color(region)
    ax.scatter(
        subset["pop_census"],
        subset["pop_worldpop"],
        c=color,
        label=region,
        s=12,
        alpha=0.7,
        edgecolors="none",
    )

# 1:1 line
lims = [1e3, merged[["pop_census", "pop_worldpop"]].max().max() * 1.2]
ax.plot(lims, lims, "k--", lw=0.8, zorder=0)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Census 2022 Projection")
ax.set_ylabel("WorldPop 2022 Estimate")
ax.set_aspect("equal")
ax.legend(fontsize=7, loc="upper left", frameon=False, ncol=2)

plt.tight_layout()
plt.savefig("figures/population_comparison.pdf")
print("Saved figures/pdf/population_comparison.pdf")
