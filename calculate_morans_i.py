import pandas as pd
import geopandas as gpd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

dhis2_levels = [
    (
        "DHIS2 Regional",
        "estimates/regional_calibrated_coverage_geojson.csv",
        "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin1.geojson",
        "adm1_name",
        "region",
        "estimates/region_neighbors.csv",
    ),
    (
        "DHIS2 Zonal",
        "estimates/zonal_calibrated_coverage_geojson.csv",
        "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin2.geojson",
        "adm2_name",
        "zone",
        "estimates/zone_neighbors.csv",
    ),
    (
        "DHIS2 Woreda",
        "estimates/woreda_calibrated_coverage_geojson.csv",
        "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin3.geojson",
        "adm3_name",
        "woreda",
        "estimates/woreda_neighbors.csv",
    ),
]

dhs_levels = [
    (
        "DHS Regional",
        "estimates/dhs_regional_estimates.csv",
        "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin1.geojson",
        "adm1_name",
        "region",
        "estimates/region_neighbors.csv",
    ),
    (
        "DHS Zonal",
        "estimates/dhs_zonal_estimates.csv",
        "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin2.geojson",
        "adm2_name",
        "zone",
        "estimates/zone_neighbors.csv",
    ),
    (
        "DHS Woreda",
        "estimates/dhs_woreda_estimates.csv",
        "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin3.geojson",
        "adm3_name",
        "woreda",
        "estimates/woreda_neighbors.csv",
    ),
]


def calc_and_print_morans(label, indicator, grp, cov_col, geo_id, neighbors_df):
    if len(grp) < 3:
        return
    names = grp[geo_id].values
    n = len(names)
    name_to_idx = {name_val: i for i, name_val in enumerate(names)}

    w = np.zeros((n, n))
    for _, row in neighbors_df.iterrows():
        if row["name_1"] in name_to_idx and row["name_2"] in name_to_idx:
            i = name_to_idx[row["name_1"]]
            j = name_to_idx[row["name_2"]]
            w[i, j] = 1
            w[j, i] = 1

    # Row-standardized
    w_sums = w.sum(axis=1, keepdims=True)
    w_sums[w_sums == 0] = 1  # Avoid division by zero
    w /= w_sums

    x = grp[cov_col].values
    dx = x - x.mean()  # Assuming equal pop distribution

    var = np.sum(dx**2)
    if var == 0:
        return

    morans_i = np.dot(dx, w.dot(dx)) / var
    print(f"{label:<13} | {indicator[:35]:<35} | I: {morans_i:+.3f} (n={n})")


def calculate_morans_i(
    name, cov_file, geo_file, geo_id, cov_id, neighbor_file, is_dhis2
):
    df = pd.read_csv(cov_file)
    gdf = gpd.read_file(geo_file)
    neighbors_df = pd.read_csv(neighbor_file)

    merged = gdf.merge(df, left_on=geo_id, right_on=cov_id, how="inner")

    if is_dhis2:
        merged = merged[merged["year"].between(2019, 2025)]
        merged = merged.dropna(subset=["calibrated_coverage"])

        avg_df = (
            merged.groupby([geo_id, "indicator"])
            .agg({"calibrated_coverage": "mean"})
            .reset_index()
        )

        for ind, grp in avg_df.groupby("indicator"):
            calc_and_print_morans(
                "2019-2025 Avg", ind, grp, "calibrated_coverage", geo_id, neighbors_df
            )
    else:
        # 2016-2019 Average
        avg_subset = merged[merged["year"].isin([2016, 2019])].dropna(
            subset=["coverage"]
        )
        avg_df = (
            avg_subset.groupby([geo_id, "indicator"])
            .agg({"coverage": "mean"})
            .reset_index()
        )

        for ind, grp in avg_df.groupby("indicator"):
            calc_and_print_morans(
                "2016-2019 Avg", ind, grp, "coverage", geo_id, neighbors_df
            )


for args in dhis2_levels:
    calculate_morans_i(*args, is_dhis2=True)

for args in dhs_levels:
    calculate_morans_i(*args, is_dhis2=False)
