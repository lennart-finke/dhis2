"""
Compute coverage estimates from PMA panel data at regional, zonal, and woreda levels.

Outputs:
  - estimates/pma_regional_estimates.csv: Regional-level estimates
  - estimates/pma_zonal_estimates.csv: Zone-level estimates
  - estimates/pma_woreda_estimates.csv: Woreda-level estimates (from GPS spatial join)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# GPS files by year
GPS_FILES = {
    2019: "data/pma/gps/PMA_ET_GPS_v1_01Nov2020.csv",
    2021: "data/pma/gps/PMAET_2021_GPS_v1_25Jul2025.csv",
    2023: "data/pma/gps/PMAET_2023_GPS_v1_14Aug2025.csv",
}

REGION_MAP = {
    1: "Tigray",
    2: "Afar",
    3: "Amhara",
    4: "Oromiya",
    5: "Somali",
    6: "Benishangul-Gumuz",
    7: "SNNP",
    8: "Gambella",
    9: "Harari",
    10: "Addis Ababa",
    11: "Dire Dawa",
    12: "Sidama",
}

# Admin boundary files
ADMIN1_FILE = "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin1.geojson"
ADMIN2_FILE = "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin2.geojson"
ADMIN3_FILE = "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin3.geojson"


def load_gps_data():
    """Load and merge all GPS files, returning GeoDataFrame with EA locations."""
    all_gps = []
    for year, path in GPS_FILES.items():
        df = pd.read_csv(path)
        df["gps_year"] = year
        all_gps.append(df)

    gps = pd.concat(all_gps, ignore_index=True)

    # Create geometry from coordinates
    geometry = [Point(lon, lat) for lon, lat in zip(gps["GPSLONG"], gps["GPSLAT"])]
    gps_gdf = gpd.GeoDataFrame(gps, geometry=geometry, crs="EPSG:4326")

    return gps_gdf


def spatial_join_admin(gps_gdf, admin1_gdf, admin2_gdf, admin3_gdf=None):
    """Spatial join GPS points to admin boundaries (regions, zones, and woredas)."""
    # Join to zones (admin2)
    joined = gpd.sjoin(
        gps_gdf,
        admin2_gdf[["adm1_name", "adm2_name", "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.rename(columns={"adm1_name": "region_geo", "adm2_name": "zone_geo"})

    # Drop index_right if exists
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    # Join to woredas (admin3) if available
    if admin3_gdf is not None:
        joined = gpd.sjoin(
            joined,
            admin3_gdf[["adm3_name", "geometry"]],
            how="left",
            predicate="within",
        )
        joined = joined.rename(columns={"adm3_name": "woreda_geo"})
        if "index_right" in joined.columns:
            joined = joined.drop(columns=["index_right"])

    return joined


def compute_coverage(series, yes_vals=[1.0], no_vals=[0.0], exclude_vals=[-88.0]):
    """Compute coverage as yes/(yes+no), excluding specified values."""
    yes_count = series.isin(yes_vals).sum()
    no_count = series.isin(no_vals).sum()
    total = yes_count + no_count
    coverage = yes_count / total if total > 0 else np.nan
    return coverage, total


def compute_pma_indicators(
    df_1yr_subset, df_6wk_subset, year, region_name, zone_name, woreda_name=None
):
    """Compute all PMA indicators for a data subset.

    Returns list of result records with region, zone, and optionally woreda.
    """
    results = []

    base_record = {"region": region_name, "zone": zone_name, "year": year}
    if woreda_name is not None:
        base_record["woreda"] = woreda_name

    # Measles first dose
    if not df_1yr_subset.empty:
        card_cov, card_n = compute_coverage(
            df_1yr_subset["baby1_card_measles1"], yes_vals=[1.0, -88.0], no_vals=[0.0]
        )
        nocard_cov, nocard_n = compute_coverage(
            df_1yr_subset["baby1_nocard_measles_yn"], yes_vals=[1.0], no_vals=[0.0]
        )
        card_yes = card_cov * card_n if card_n > 0 else 0
        nocard_yes = nocard_cov * nocard_n if nocard_n > 0 else 0
        total_n = card_n + nocard_n
        results.append(
            {
                **base_record,
                "indicator": "0-12 months Measles1",
                "coverage": (card_yes + nocard_yes) / total_n
                if total_n > 0
                else np.nan,
                "denominator": total_n,
            }
        )

        # Pentavalent third dose
        card_cov, card_n = compute_coverage(
            df_1yr_subset["baby1_card_pentavalent3"],
            yes_vals=[1.0, -88.0],
            no_vals=[0.0],
        )
        nocard_yes = (df_1yr_subset["baby1_nocard_pentavalent_ct"] >= 3.0).sum()
        nocard_total = df_1yr_subset["baby1_nocard_pentavalent_yn"].isin(
            [1.0, -88.0, 0.0]
        )
        nocard_n = nocard_total.sum()
        # Ensure we don't count something outside the denominator

        card_yes_count = card_cov * card_n if card_n > 0 else 0
        total_n = card_n + nocard_n
        results.append(
            {
                **base_record,
                "indicator": "0-12 months Pentavalent3",
                "coverage": (card_yes_count + nocard_yes) / total_n
                if total_n > 0
                else np.nan,
                "denominator": total_n,
            }
        )

        # Vitamin A
        card_cov, card_n = compute_coverage(
            df_1yr_subset["baby1_card_vit_a"], yes_vals=[1.0, -88.0], no_vals=[0.0]
        )
        nocard_cov, nocard_n = compute_coverage(
            df_1yr_subset["baby1_nocard_vit_a_yn"], yes_vals=[1.0], no_vals=[0.0]
        )
        card_yes = card_cov * card_n if card_n > 0 else 0
        nocard_yes = nocard_cov * nocard_n if nocard_n > 0 else 0
        total_n = card_n + nocard_n
        results.append(
            {
                **base_record,
                "indicator": "0-12 months Vitamin A",
                "coverage": (card_yes + nocard_yes) / total_n
                if total_n > 0
                else np.nan,
                "denominator": total_n,
            }
        )

    if not df_6wk_subset.empty:
        # Deworming
        deworm_cov, deworm_n = compute_coverage(
            df_6wk_subset["anc_wormsmed"], yes_vals=[1.0], no_vals=[0.0]
        )
        results.append(
            {
                **base_record,
                "indicator": "Mother dewormed during pregnancy",
                "coverage": deworm_cov,
                "denominator": deworm_n,
            }
        )

        # ANC first visit
        df_6wk_copy = df_6wk_subset.copy()
        df_6wk_copy["anc_yn"] = df_6wk_copy[["anc_phcp_yn_pp", "anc_hew_yn_pp"]].max(
            axis=1
        )
        anc_cov, anc_n = compute_coverage(
            df_6wk_copy["anc_yn"], yes_vals=[1.0], no_vals=[0.0]
        )
        results.append(
            {
                **base_record,
                "indicator": "Post-partum had ANC",
                "coverage": anc_cov,
                "denominator": anc_n,
            }
        )

        # Skilled birth attendance
        df_6wk_copy["skilled_ba"] = df_6wk_copy["deliv_assit"].map(
            {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 96: 0}
        )
        sba_cov, sba_n = compute_coverage(
            df_6wk_copy["skilled_ba"], yes_vals=[1.0], no_vals=[0.0]
        )
        results.append(
            {
                **base_record,
                "indicator": "SBA",
                "coverage": sba_cov,
                "denominator": sba_n,
            }
        )

        # C-section
        if "deliv_csection" in df_6wk_copy.columns:
            # deliv_csection: 1.0 = yes (C-section), 0.0 = no
            c_count = (df_6wk_copy["deliv_csection"] == 1.0).sum()
            df_6wk_copy["skilled_ba_calc"] = df_6wk_copy["deliv_assit"].map(
                {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 96: 0}
            )
            sba_yes = (df_6wk_copy["skilled_ba_calc"] == 1).sum()
            if sba_yes > 0:
                results.append(
                    {
                        **base_record,
                        "indicator": "C-section",
                        "coverage": c_count / sba_yes,
                        "denominator": sba_yes,
                    }
                )

    return results


def process_cohort(cohort_num, year, gps_joined):
    """Process one cohort and return tuple of (zonal_results, woreda_results).

    Args:
        cohort_num: Cohort number (1 or 2)
        year: Survey year
        gps_joined: GPS joined DataFrame
    """
    # Load data
    if cohort_num == 1:
        df_1yr = pd.read_csv(
            "data/pma/household/PMAET_Panel_Cohort1_1yrFU_v3.0_1Sep2024.csv",
            low_memory=False,
        )
        df_6wk = pd.read_csv(
            "data/pma/household/PMAET_HQFQ_Panel_Cohort1_6wkFU_v3.0_1Sep2024.csv",
            low_memory=False,
        )
    else:
        df_1yr = pd.read_csv(
            "data/pma/household/PMAET_HQFQ_Panel_Cohort2_1yrFU_v2.0_17Apr2024.csv",
            low_memory=False,
        )
        df_6wk = pd.read_csv(
            "data/pma/household/PMAET_HQFQ_Panel_Cohort2_6wkFU_v2.0_28Oct2023.csv",
            low_memory=False,
        )

    # Get GPS lookup for this year's EAs (including woreda if available)
    gps_year = 2019 if cohort_num == 1 else 2021
    gps_cols = ["EA_ID", "region_geo", "zone_geo"]
    if "woreda_geo" in gps_joined.columns:
        gps_cols.append("woreda_geo")
    gps_subset = gps_joined[gps_joined["gps_year"] == gps_year][
        gps_cols
    ].drop_duplicates("EA_ID")

    # Merge GPS info to survey data
    df_1yr = df_1yr.merge(gps_subset, on="EA_ID", how="left")
    df_6wk = df_6wk.merge(gps_subset, on="EA_ID", how="left")

    if "region" in df_1yr.columns:
        df_1yr["region_name"] = df_1yr["region"].map(REGION_MAP)
    else:
        df_1yr["region_name"] = df_1yr["region_geo"]
    if "region" in df_6wk.columns:
        df_6wk["region_name"] = df_6wk["region"].map(REGION_MAP)
    else:
        df_6wk["region_name"] = df_6wk["region_geo"]

    zonal_results = []
    woreda_results = []

    # Check if woreda data is available
    has_woreda = "woreda_geo" in df_1yr.columns and df_1yr["woreda_geo"].notna().any()

    # Process at zonal level
    zones = df_1yr[["region_name", "zone_geo"]].drop_duplicates().dropna()

    for _, row in zones.iterrows():
        region_name = row["region_name"]
        zone_name = row["zone_geo"]

        df_1yr_zone = df_1yr[
            (df_1yr["region_name"] == region_name) & (df_1yr["zone_geo"] == zone_name)
        ]
        df_6wk_zone = df_6wk[
            (df_6wk["region_name"] == region_name) & (df_6wk["zone_geo"] == zone_name)
        ]

        if df_1yr_zone.empty and df_6wk_zone.empty:
            continue

        zonal_results.extend(
            compute_pma_indicators(
                df_1yr_zone, df_6wk_zone, year, region_name, zone_name
            )
        )

    # Process at woreda level if available
    if has_woreda:
        woredas = (
            df_1yr[["region_name", "zone_geo", "woreda_geo"]].drop_duplicates().dropna()
        )

        for _, row in woredas.iterrows():
            region_name = row["region_name"]
            zone_name = row["zone_geo"]
            woreda_name = row["woreda_geo"]

            df_1yr_woreda = df_1yr[
                (df_1yr["region_name"] == region_name)
                & (df_1yr["zone_geo"] == zone_name)
                & (df_1yr["woreda_geo"] == woreda_name)
            ]
            df_6wk_woreda = df_6wk[
                (df_6wk["region_name"] == region_name)
                & (df_6wk["zone_geo"] == zone_name)
                & (df_6wk["woreda_geo"] == woreda_name)
            ]

            if df_1yr_woreda.empty and df_6wk_woreda.empty:
                continue

            woreda_results.extend(
                compute_pma_indicators(
                    df_1yr_woreda,
                    df_6wk_woreda,
                    year,
                    region_name,
                    zone_name,
                    woreda_name,
                )
            )

    return zonal_results, woreda_results


def aggregate_to_regional(zonal_df):
    """Aggregate zonal estimates to regional level (weighted by denominator)."""
    regional_rows = []
    for (region, year, indicator), group in zonal_df.groupby(
        ["region", "year", "indicator"]
    ):
        valid = group.dropna(subset=["coverage", "denominator"])
        if valid.empty or valid["denominator"].sum() == 0:
            continue
        weighted_cov = (valid["coverage"] * valid["denominator"]).sum() / valid[
            "denominator"
        ].sum()
        regional_rows.append(
            {
                "region": region,
                "year": year,
                "indicator": indicator,
                "coverage": weighted_cov,
                "denominator": valid["denominator"].sum(),
            }
        )
    return pd.DataFrame(regional_rows)


def main():
    print("Using PMA region variable for region assignment")
    print("Zone/woreda assignment uses GPS coordinates")
    print()

    # Load GPS data and admin boundaries
    print("Loading GPS data...")
    gps_gdf = load_gps_data()
    print(f"  Loaded {len(gps_gdf)} GPS points")

    print("Loading admin boundaries...")
    admin1 = gpd.read_file(ADMIN1_FILE)
    admin2 = gpd.read_file(ADMIN2_FILE)
    admin3 = gpd.read_file(ADMIN3_FILE)
    print(f"  Admin3 (woredas): {len(admin3)} boundaries")

    print("Performing spatial join...")
    gps_joined = spatial_join_admin(gps_gdf, admin1, admin2, admin3)

    # Convert to DataFrame for merging
    gps_joined_df = pd.DataFrame(gps_joined.drop(columns=["geometry"]))
    if "woreda_geo" in gps_joined_df.columns:
        print(f"  EAs with woreda: {gps_joined_df['woreda_geo'].notna().sum()}")

    # Process cohorts
    all_zonal_results = []
    all_woreda_results = []

    print("Processing cohort 1 (year 2020)...")
    zonal, woreda = process_cohort(1, 2020, gps_joined_df)
    all_zonal_results.extend(zonal)
    all_woreda_results.extend(woreda)
    print(f"  Generated {len(zonal)} zonal, {len(woreda)} woreda records")

    print("Processing cohort 2 (year 2022)...")
    zonal, woreda = process_cohort(2, 2022, gps_joined_df)
    all_zonal_results.extend(zonal)
    all_woreda_results.extend(woreda)
    print(f"  Generated {len(zonal)} zonal, {len(woreda)} woreda records")

    # Create zonal DataFrame
    zonal_df = pd.DataFrame(all_zonal_results)
    zonal_df = zonal_df.sort_values(["indicator", "region", "zone", "year"])

    # Save zonal estimates
    zonal_out = Path("estimates/pma_zonal_estimates.csv")
    zonal_df.to_csv(zonal_out, index=False)
    print(f"Wrote {len(zonal_df):,} zonal rows to {zonal_out}")

    # Create and save woreda DataFrame
    if all_woreda_results:
        woreda_df = pd.DataFrame(all_woreda_results)
        woreda_df = woreda_df.sort_values(
            ["indicator", "region", "zone", "woreda", "year"]
        )
        woreda_out = Path("estimates/pma_woreda_estimates.csv")
        woreda_df.to_csv(woreda_out, index=False)
        print(f"Wrote {len(woreda_df):,} woreda rows to {woreda_out}")
        print(f"  {woreda_df['woreda'].nunique()} unique woredas")

    # Aggregate to regional
    regional_df = aggregate_to_regional(zonal_df)

    # Add national aggregates
    def weighted_mean(group):
        if group["denominator"].sum() > 0:
            return (group["coverage"] * group["denominator"]).sum() / group[
                "denominator"
            ].sum()
        return np.nan

    national = regional_df.groupby(["year", "indicator"], as_index=False).apply(
        lambda g: pd.Series(
            {"coverage": weighted_mean(g), "denominator": g["denominator"].sum()}
        ),
        include_groups=False,
    )
    national["region"] = "National"
    regional_df = pd.concat([regional_df, national], ignore_index=True)
    regional_df = regional_df.sort_values(["indicator", "region", "year"])

    # Save regional estimates
    regional_out = Path("estimates/pma_regional_estimates.csv")
    regional_df.to_csv(regional_out, index=False)
    print(f"Wrote {len(regional_df):,} regional rows to {regional_out}")
    print(
        f"  {regional_df['indicator'].nunique()} indicators, {regional_df['region'].nunique()} regions"
    )


if __name__ == "__main__":
    main()
