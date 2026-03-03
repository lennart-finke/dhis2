"""
Compute coverage estimates from DHS survey data at regional, zonal, and woreda levels.

Outputs:
  - estimates/dhs_regional_estimates.csv: Regional-level estimates
  - estimates/dhs_zonal_estimates.csv: Zone-level estimates
  - estimates/dhs_woreda_estimates.csv: Woreda-level estimates (from GPS spatial join)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pyreadstat
import geopandas as gpd
import re

# DHS surveys to process
DHS_FILES = {
    2016: "data/dhs/ET_2016_DHS_11092025_2313_234201/ETKR71DT/ETKR71FL.DTA",
    2019: "data/dhs/ET_2019_INTERIMDHS_11092025_2313_234201/ETKR81DT/ETKR81FL.DTA",
}

# GPS files for each survey
GPS_FILES = {
    2016: "data/dhs/ET_2016_DHS_11092025_2313_234201/ETGE71FL/ETGE71FL.shp",
    2019: "data/dhs/ET_2019_INTERIMDHS_11092025_2313_234201/ETGE81FL/ETGE81FL.shp",
}

# Admin boundary files
ADMIN1_FILE = "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin1.geojson"
ADMIN2_FILE = "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin2.geojson"
ADMIN3_FILE = "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin3.geojson"

# Manual mapping for DHIS2-DHS region name discrepancies
REGION_NAME_CORRECTIONS = {
    "addis adaba": "Addis Ababa",
    "benishangul": "Benishangul Gumuz",
    "benishangul gumz": "Benishangul Gumuz",
    "benishangul-gumuz": "Benishangul Gumuz",
    "gambella": "Gambela",
    "tigrai": "Tigray",
    "snnpr": "SNNP",
    "snnp": "SNNP",
}

# DHS v024 region code to name mapping
DHS_REGION_MAP = {
    1: "Tigray",
    2: "Afar",
    3: "Amhara",
    4: "Oromia",
    5: "Somali",
    6: "Benishangul Gumuz",
    7: "SNNP",
    8: "Gambela",
    9: "Harari",
    10: "Addis Ababa",
    11: "Dire Dawa",
}


def normalize_region_name(name):
    """Normalize DHS region name for consistency."""
    name = name.lower()
    name = re.sub(
        r" (region|city administration|regional health bureau|peoples|people)$",
        "",
        name,
    )
    name = name.strip()
    return REGION_NAME_CORRECTIONS.get(name, name.title())


def load_gps_and_join_admin(gps_path, admin1, admin2, admin3=None):
    """Load GPS shapefile and spatial join to admin boundaries (region, zone, woreda)."""
    gps = gpd.read_file(gps_path)

    # Filter out points with no coordinates (LATNUM=0)
    gps = gps[gps["LATNUM"] != 0].copy()

    # Ensure CRS matches
    if gps.crs != admin2.crs:
        gps = gps.to_crs(admin2.crs)

    # Spatial join to zones (admin2)
    joined = gpd.sjoin(
        gps,
        admin2[["adm1_name", "adm2_name", "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.rename(columns={"adm1_name": "region_geo", "adm2_name": "zone_geo"})

    # Drop index_right to allow second join
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    # Spatial join to woredas (admin3) if available
    if admin3 is not None:
        # Need to convert back to GeoDataFrame for second join
        joined_gdf = gpd.GeoDataFrame(joined, geometry="geometry", crs=admin2.crs)
        joined = gpd.sjoin(
            joined_gdf,
            admin3[["adm3_name", "geometry"]],
            how="left",
            predicate="within",
        )
        joined = joined.rename(columns={"adm3_name": "woreda_geo"})
        if "index_right" in joined.columns:
            joined = joined.drop(columns=["index_right"])

        # Create lookup by cluster with woreda
        cluster_lookup = joined[
            ["DHSCLUST", "region_geo", "zone_geo", "woreda_geo"]
        ].drop_duplicates("DHSCLUST")
    else:
        # Create lookup by cluster without woreda
        cluster_lookup = joined[["DHSCLUST", "region_geo", "zone_geo"]].drop_duplicates(
            "DHSCLUST"
        )

    cluster_lookup["DHSCLUST"] = cluster_lookup["DHSCLUST"].astype(int)

    return cluster_lookup


def compute_malnutrition(df):
    """Compute malnutrition screening rate for children 0-59 months."""
    df_filtered = df[(df["b19"] >= 0) & (df["b19"] <= 59) & (df["hw72"].notna())].copy()
    if df_filtered.empty:
        return None, 0
    screened_rate = len(df_filtered) / len(df)
    return screened_rate, len(df)


def compute_anc_sba(df):
    """Compute ANC and SBA coverage."""
    results = {}
    denominators = {}

    # ANC
    anc_valid = df[df["m14"].notna() & (df["m14"] >= 0)]
    if not anc_valid.empty:
        results["anc"] = (anc_valid["m14"] >= 1).sum() / len(anc_valid)
        denominators["anc"] = len(anc_valid)

    # SBA
    delivery_valid = df[df[["m3a", "m3b", "m3c"]].notna().any(axis=1)].copy()
    if not delivery_valid.empty:
        skilled = (
            (delivery_valid["m3a"] == 1)
            | (delivery_valid["m3b"] == 1)
            | (delivery_valid["m3c"] == 1)
        )
        results["sba"] = skilled.sum() / len(delivery_valid)
        denominators["sba"] = len(delivery_valid)

    return results, denominators


def compute_vaccinations(df):
    """Compute vaccination rates for children 12-23 months."""
    results = {}
    df_filtered = df[(df["b19"] >= 12) & (df["b19"] <= 23)].copy()
    total = len(df_filtered)
    if total == 0:
        return {}, 0

    measles_vax = df_filtered["h9"].isin([1, 2, 3])
    results["measles1"] = measles_vax.sum() / total

    dpt3_vax = df_filtered["h7"].isin([1, 2, 3])
    results["dpt3"] = dpt3_vax.sum() / total

    return results, total


def compute_deworming(df):
    """Compute deworming coverage for children 24-59 months."""
    df_filtered = df[(df["b19"] >= 24) & (df["b19"] <= 59)].copy()
    if df_filtered.empty:
        return None, 0
    total = len(df_filtered)
    dewormed = df_filtered["h43"] == 1
    return {"dewormed": dewormed.sum() / total}, total


def compute_wasting(df):
    """Compute wasting prevalence for children 0-59 months."""
    df_filtered = df[
        (df["b19"] >= 0)
        & (df["b19"] <= 59)
        & (df["hw72"].notna())
        & (df["hw72"] > -600)
        & (df["hw72"] < 600)
    ].copy()
    if df_filtered.empty:
        return None, 0
    wasted = (df_filtered["hw72"] < -200) & (df_filtered["hw72"] > -300)
    return wasted.sum() / len(df_filtered), len(df_filtered)


def compute_lbw(df):
    """Compute low birth weight prevalence."""
    valid = df[(df["m19"].notna()) & (df["m19"] > 0) & (df["m19"] < 9000)]
    if len(valid) == 0:
        return None, 0
    if valid["m19"].max() > 100:
        lbw_count = (valid["m19"] < 2500).sum()
    else:
        lbw_count = (valid["m19"] < 2.5).sum()
    return lbw_count / len(valid), len(valid)


def compute_c_section(df):
    """Compute C-section rate (C-sections / SBA)."""
    required_cols = ["m17", "m3a", "m3b", "m3c"]
    if not all(c in df.columns for c in required_cols):
        return None, 0

    sba_valid = df[df[["m3a", "m3b", "m3c"]].notna().any(axis=1)]
    sba_count = (
        (sba_valid["m3a"] == 1) | (sba_valid["m3b"] == 1) | (sba_valid["m3c"] == 1)
    ).sum()

    if sba_count == 0:
        return 0, 0

    c_section_count = (df["m17"] == 1).sum()
    return c_section_count / sba_count, sba_count


def compute_indicators_for_subset(
    df_subset, year, region_name, zone_name, woreda_name=None
):
    """Compute all indicators for a data subset.

    Returns list of result records with region, zone, and optionally woreda.
    """
    results = []

    base_record = {"region": region_name, "zone": zone_name, "year": year}
    if woreda_name is not None:
        base_record["woreda"] = woreda_name

    # Malnutrition screening
    screened_rate, denom_mal = compute_malnutrition(df_subset)
    if screened_rate is not None:
        results.append(
            {
                **base_record,
                "indicator": "Children <5 Years Screened for Malnutrition",
                "coverage": screened_rate,
                "denominator": denom_mal,
            }
        )

    # ANC and SBA
    anc_sba_rates, anc_sba_denoms = compute_anc_sba(df_subset)
    if "anc" in anc_sba_rates:
        results.append(
            {
                **base_record,
                "indicator": "ANC First Visit",
                "coverage": anc_sba_rates["anc"],
                "denominator": anc_sba_denoms["anc"],
            }
        )
    if "sba" in anc_sba_rates:
        results.append(
            {
                **base_record,
                "indicator": "Skilled Birth Attendance",
                "coverage": anc_sba_rates["sba"],
                "denominator": anc_sba_denoms["sba"],
            }
        )

    # Vaccination
    vax, vax_denom = compute_vaccinations(df_subset)
    if "measles1" in vax:
        results.append(
            {
                **base_record,
                "indicator": "Measles 1st Dose",
                "coverage": vax["measles1"],
                "denominator": vax_denom,
            }
        )
    if "dpt3" in vax:
        results.append(
            {
                **base_record,
                "indicator": "DPT3/Penta3",
                "coverage": vax["dpt3"],
                "denominator": vax_denom,
            }
        )

    # Deworming (2016 only)
    if year == 2016:
        deworm, deworm_denom = compute_deworming(df_subset)
        if deworm is not None and "dewormed" in deworm:
            results.append(
                {
                    **base_record,
                    "indicator": "Drugs for Intestinal Parasites",
                    "coverage": deworm["dewormed"],
                    "denominator": deworm_denom,
                }
            )

        # Low birth weight
        lbw_rate, lbw_denom = compute_lbw(df_subset)
        if lbw_rate is not None:
            results.append(
                {
                    **base_record,
                    "indicator": "Low Birth Weight",
                    "coverage": lbw_rate,
                    "denominator": lbw_denom,
                }
            )

    # MAM (Moderate Acute Malnutrition)
    wasting_rate, wasting_denom = compute_wasting(df_subset)
    if wasting_rate is not None:
        results.append(
            {
                **base_record,
                "indicator": "MAM",
                "coverage": wasting_rate,
                "denominator": wasting_denom,
            }
        )

    # C-section
    c_sec_rate, c_sec_denom = compute_c_section(df_subset)
    if c_sec_rate is not None and c_sec_denom > 0:
        results.append(
            {
                **base_record,
                "indicator": "C-section",
                "coverage": c_sec_rate,
                "denominator": c_sec_denom,
            }
        )

    return results


def process_dhs_survey(year, file_path, cluster_lookup):
    """Process one DHS survey and return tuple of (zonal_results, woreda_results).

    Args:
        year: Survey year
        file_path: Path to DHS data file
        cluster_lookup: GPS cluster lookup DataFrame (must include woreda_geo if woreda level needed)
    """
    df, meta = pyreadstat.read_dta(file_path)

    # Merge cluster lookup to get zone/woreda from GPS
    df["v001"] = df["v001"].astype(int)
    df = df.merge(cluster_lookup, left_on="v001", right_on="DHSCLUST", how="left")

    df["region_name"] = df["v024"].map(DHS_REGION_MAP)

    # CRITICAL FIX: Harmonize region_geo from GPS to match survey region names
    # The GeoJSON uses "Benishangul Gumz" but survey uses "Benishangul Gumuz"
    # Without this, GPS-derived zones won't match to survey-derived regions
    if "region_geo" in df.columns:
        gps_to_survey_region_map = {
            "Benishangul Gumz": "Benishangul Gumuz",
            # Add other mappings if needed in the future
        }
        df["region_geo"] = df["region_geo"].replace(gps_to_survey_region_map)

    zonal_results = []
    woreda_results = []

    # Check if woreda data is available
    has_woreda = "woreda_geo" in df.columns and df["woreda_geo"].notna().any()

    # Process at zonal level
    # CRITICAL: Only use zone assignments where GPS region matches survey region
    # DHS sometimes assigns cluster IDs to regions that don't match GPS coordinates
    # So we must filter to ensure region_name (from survey) matches region_geo (from GPS)
    df_valid_zones = df[df["region_name"] == df["region_geo"]].copy()
    zones = df_valid_zones[["region_name", "zone_geo"]].drop_duplicates().dropna()

    print(f"  Processing {len(zones)} unique (region, zone) combinations")
    print(
        f"  Filtered out {len(df) - len(df_valid_zones)} observations with region mismatch"
    )

    for _, row in zones.iterrows():
        region_name = row["region_name"]
        zone_name = row["zone_geo"]

        if region_name is None:
            continue

        # Use df_valid_zones instead of df to ensure we only get observations
        # where the survey region matches the GPS region
        df_zone = df_valid_zones[
            (df_valid_zones["region_name"] == region_name)
            & (df_valid_zones["zone_geo"] == zone_name)
        ]

        if df_zone.empty:
            continue

        # Compute zonal-level indicators
        zonal_results.extend(
            compute_indicators_for_subset(df_zone, year, region_name, zone_name)
        )

    # Process at woreda level if available
    if has_woreda:
        # Use the same filtered dataset to ensure region consistency
        woredas = (
            df_valid_zones[["region_name", "zone_geo", "woreda_geo"]]
            .drop_duplicates()
            .dropna()
        )

        for _, row in woredas.iterrows():
            region_name = row["region_name"]
            zone_name = row["zone_geo"]
            woreda_name = row["woreda_geo"]

            if region_name is None:
                continue

            df_woreda = df_valid_zones[
                (df_valid_zones["region_name"] == region_name)
                & (df_valid_zones["zone_geo"] == zone_name)
                & (df_valid_zones["woreda_geo"] == woreda_name)
            ]

            if df_woreda.empty:
                continue

            # Compute woreda-level indicators
            woreda_results.extend(
                compute_indicators_for_subset(
                    df_woreda, year, region_name, zone_name, woreda_name
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


def aggregate_woreda_to_zonal(woreda_df):
    """Aggregate woreda estimates to zonal level (weighted by denominator)."""
    zonal_rows = []
    for (region, zone, year, indicator), group in woreda_df.groupby(
        ["region", "zone", "year", "indicator"]
    ):
        valid = group.dropna(subset=["coverage", "denominator"])
        if valid.empty or valid["denominator"].sum() == 0:
            continue
        weighted_cov = (valid["coverage"] * valid["denominator"]).sum() / valid[
            "denominator"
        ].sum()
        zonal_rows.append(
            {
                "region": region,
                "zone": zone,
                "year": year,
                "indicator": indicator,
                "coverage": weighted_cov,
                "denominator": valid["denominator"].sum(),
            }
        )
    return pd.DataFrame(zonal_rows)


def load_2025_estimates():
    """Load and process 2025 DHS estimates from parsed tables."""
    results = []

    # SNNP sub-regions to aggregate
    snnp_regions = {
        "Central Ethiopia",
        "Sidama",
        "South West Ethiopia",
        "South Ethiopia",
    }

    def get_mapped_region(r_name):
        norm = normalize_region_name(r_name)
        if norm in snnp_regions:
            return "SNNP"
        return norm

    # Process Table 9: SBA (skilled_provider2 / n_births)
    # Note: skilled_provider2 is percentage in table
    df9 = pd.read_csv("data/dhs/2025/table_9.csv")
    for _, row in df9.iterrows():
        region = get_mapped_region(row["Region"])
        # SBA Coverage
        sba_cov = row["skilled_provider2"] / 100.0
        denom = row["n_births"]
        results.append(
            {
                "region": region,
                "year": 2025,
                "indicator": "Skilled Birth Attendance",
                "coverage": sba_cov,
                "denominator": denom,
            }
        )

    # Process Table 10: DPT3, Measles 1st Dose
    # Denom: N_Children for both
    df10 = pd.read_csv("data/dhs/2025/table_10.csv")
    for _, row in df10.iterrows():
        region = get_mapped_region(row["Region"])
        denom = row["N_Children"]

        # DPT3
        results.append(
            {
                "region": region,
                "year": 2025,
                "indicator": "DPT3/Penta3",
                "coverage": row["DPT3"] / 100.0,
                "denominator": denom,
            }
        )

        # Measles
        results.append(
            {
                "region": region,
                "year": 2025,
                "indicator": "Measles 1st Dose",
                "coverage": row["MCV1"] / 100.0,
                "denominator": denom,
            }
        )

    # Process Table 12: MAM (weight_below_2sd - weight_below_3sd)
    # Denom: n_children_weight
    df12 = pd.read_csv("data/dhs/2025/table_12.csv")
    for _, row in df12.iterrows():
        region = get_mapped_region(row["Region"])
        denom = row["n_children_weight"]

        # MAM (Moderate Acute Malnutrition)
        mam_val = row["weight_below_2sd"] - row["weight_below_3sd"]
        # Ensure non-negative due to float precision or data quirks
        mam_val = max(0, mam_val)

        results.append(
            {
                "region": region,
                "year": 2025,
                "indicator": "MAM",
                "coverage": mam_val / 100.0,
                "denominator": denom,
            }
        )

    if not results:
        return pd.DataFrame()

    # Aggregate by region (this handles collapsing SNNP sub-regions)
    df = pd.DataFrame(results)

    aggregated_rows = []
    for (region, year, indicator), group in df.groupby(["region", "year", "indicator"]):
        valid = group.dropna(subset=["coverage", "denominator"])
        total_denom = valid["denominator"].sum()

        if total_denom == 0:
            continue

        weighted_cov = (valid["coverage"] * valid["denominator"]).sum() / total_denom

        aggregated_rows.append(
            {
                "region": region,
                "year": year,
                "indicator": indicator,
                "coverage": weighted_cov,
                "denominator": total_denom,
            }
        )

    return pd.DataFrame(aggregated_rows)


def main():
    print("Using DHS v024 region variable for region assignment")
    print("Zone/woreda assignment uses GPS coordinates")
    print()

    # Load admin boundaries
    print("Loading admin boundaries...")
    admin1 = gpd.read_file(ADMIN1_FILE)
    admin2 = gpd.read_file(ADMIN2_FILE)
    admin3 = gpd.read_file(ADMIN3_FILE)
    print(f"  Admin3 (woredas): {len(admin3)} boundaries")

    all_zonal_results = []
    all_woreda_results = []

    for year, file_path in DHS_FILES.items():
        print(f"Processing DHS {year}...")

        # Load GPS and create cluster lookup with woreda
        gps_path = GPS_FILES[year]
        cluster_lookup = load_gps_and_join_admin(gps_path, admin1, admin2, admin3)
        print(f"  GPS clusters mapped: {len(cluster_lookup)}")
        if "woreda_geo" in cluster_lookup.columns:
            print(
                f"  Clusters with woreda: {cluster_lookup['woreda_geo'].notna().sum()}"
            )

        # Process survey
        zonal_results, woreda_results = process_dhs_survey(
            year, file_path, cluster_lookup
        )
        all_zonal_results.extend(zonal_results)
        all_woreda_results.extend(woreda_results)
        print(
            f"  Generated {len(zonal_results)} zonal, {len(woreda_results)} woreda records"
        )

    # Create zonal DataFrame
    zonal_df = pd.DataFrame(all_zonal_results)
    zonal_df = zonal_df.sort_values(["indicator", "region", "zone", "year"])

    # Save zonal estimates
    zonal_out = Path("estimates/dhs_zonal_estimates.csv")
    zonal_df.to_csv(zonal_out, index=False)
    print(f"Wrote {len(zonal_df):,} zonal rows to {zonal_out}")

    # Create and save woreda DataFrame
    if all_woreda_results:
        woreda_df = pd.DataFrame(all_woreda_results)
        woreda_df = woreda_df.sort_values(
            ["indicator", "region", "zone", "woreda", "year"]
        )
        woreda_out = Path("estimates/dhs_woreda_estimates.csv")
        woreda_df.to_csv(woreda_out, index=False)
        print(f"Wrote {len(woreda_df):,} woreda rows to {woreda_out}")
        print(f"  {woreda_df['woreda'].nunique()} unique woredas")

    # Aggregate to regional
    regional_df = aggregate_to_regional(zonal_df)

    # Supplement with 2025 estimates
    print("Loading 2025 DHS estimates...")
    regional_2025 = load_2025_estimates()
    if not regional_2025.empty:
        regional_df = pd.concat([regional_df, regional_2025], ignore_index=True)
        print(f"  Added {len(regional_2025)} records from 2025 tables")

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
    regional_out = Path("estimates/dhs_regional_estimates.csv")
    regional_df.to_csv(regional_out, index=False)
    print(f"Wrote {len(regional_df):,} regional rows to {regional_out}")
    print(
        f"  {regional_df['indicator'].nunique()} indicators, {regional_df['region'].nunique()} regions"
    )


if __name__ == "__main__":
    main()
