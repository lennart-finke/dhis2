"""
Compute coverage estimates for key DHIS2 indicators using GPS-based boundary matching.

Uses GPS-derived administrative boundaries from facility_boundary_mapping.csv to match
DHIS2 woreda names to GeoJSON woredas for population lookup. This provides more accurate
boundary assignment than name-matching alone.

Saves to:
  - estimates/dhis2_coverage_estimates_gps.csv (regional)
  - estimates/dhis2_coverage_estimates_zonal_gps.csv
  - estimates/dhis2_coverage_estimates_woreda_gps.csv
"""

from pathlib import Path
import pandas as pd
from dhis2_utils import (
    load_population_data,
    get_regional_population,
    get_zonal_population,
    get_woreda_population,
    get_kebele_population,
    GEO_TO_NORMALIZED_REGION,
)

# Map indicators to their appropriate population fraction
INDICATOR_TO_FRACTION = {
    "EPI_<1 Year Received 3rd Dose Penta": "<1 Year",
    "EPI_Children <1 Year Measels- 1st Dose": "<1 Year",
    "NUT_Children 6-59 Months received Vitamin A by Age": "6-59 Months",
    "NUT_Children 6-59 Months Received Vitamin A by Dose": "6-59 Months",
    "NUT_Children 24 - 59 Months Dewormed": "24-59 Months",
    "Number of pregnant women who received ANC first visit by gestational week": "Births",
    "MAT_Skilled Birth Attendance": "Births",
    "MAT_Births attended by Level IV HEW And Nurses at HP": "Births",
    "NUT_Children < 5 years Screened for Acute Malnutrition": "6-59 Months",
    "NUT_Children <5 Years Screened with MAM": "6-59 Months",
    "NUT_Children <5 Years Screened with SAM": "6-59 Months",
}

# Load shared data
fractions_df = pd.read_csv("estimates/population_fractions_un.csv")
pop_df = load_population_data()


def compute_lbw_coverage(dhis_df: pd.DataFrame, unit_cols: list[str]) -> pd.DataFrame:
    """Compute LBW coverage from DHIS2 numerator/denominator."""
    lbw_num = dhis_df[dhis_df["indicator"] == "NUT_Newborns < 2500 gm"]
    lbw_den = dhis_df[dhis_df["indicator"] == "NUT_Live Births Weighed"]

    if lbw_num.empty or lbw_den.empty:
        return pd.DataFrame()

    lbw_numerator = lbw_num[unit_cols + ["year", "value"]].rename(
        columns={"value": "lbw_numerator"}
    )
    lbw_denominator = lbw_den[unit_cols + ["year", "value"]].rename(
        columns={"value": "lbw_denominator"}
    )

    merged = lbw_numerator.merge(lbw_denominator, on=unit_cols + ["year"], how="inner")
    merged = merged[merged["lbw_denominator"] > 0]
    merged["coverage"] = merged["lbw_numerator"] / merged["lbw_denominator"]
    merged["indicator"] = "Low Birth Weight"
    merged["value"] = merged["lbw_numerator"]
    merged["denominator_population"] = merged["lbw_denominator"]
    merged["age_group"] = "LBW"
    merged["fraction"] = 1.0
    merged["population"] = merged["lbw_denominator"]

    return merged[
        unit_cols
        + [
            "year",
            "indicator",
            "value",
            "population",
            "age_group",
            "fraction",
            "denominator_population",
            "coverage",
        ]
    ]


def compute_csec_coverage(dhis_df: pd.DataFrame, unit_cols: list[str]) -> pd.DataFrame:
    """Compute C-section coverage from DHIS2 numerator/denominator."""
    csec_num = dhis_df[dhis_df["indicator"] == "MAT_Births By Caesarean Section"]
    csec_den = dhis_df[dhis_df["indicator"] == "MAT_Skilled Birth Attendance"]

    if csec_num.empty or csec_den.empty:
        return pd.DataFrame()

    csec_numerator = csec_num[unit_cols + ["year", "value"]].rename(
        columns={"value": "csec_numerator"}
    )
    csec_denominator = csec_den[unit_cols + ["year", "value"]].rename(
        columns={"value": "csec_denominator"}
    )

    merged = csec_numerator.merge(
        csec_denominator, on=unit_cols + ["year"], how="inner"
    )
    merged = merged[merged["csec_denominator"] > 0]
    merged["coverage"] = merged["csec_numerator"] / merged["csec_denominator"]
    merged["indicator"] = "MAT_Births By Caesarean Section"
    merged["value"] = merged["csec_numerator"]
    merged["denominator_population"] = merged["csec_denominator"]
    merged["age_group"] = "C-section"
    merged["fraction"] = 1.0
    merged["population"] = merged["csec_denominator"]

    return merged[
        unit_cols
        + [
            "year",
            "indicator",
            "value",
            "population",
            "age_group",
            "fraction",
            "denominator_population",
            "coverage",
        ]
    ]


def compute_population_coverage(
    dhis_df: pd.DataFrame,
    pop_df: pd.DataFrame,
    unit_cols: list[str],
) -> pd.DataFrame:
    """Compute coverage for population-based indicators."""
    # Filter to population-based indicators
    exclude_indicators = [
        "NUT_Newborns < 2500 gm",
        "NUT_Live Births Weighed",
        "MAT_Births By Caesarean Section",
    ]
    dhis_filtered = dhis_df[~dhis_df["indicator"].isin(exclude_indicators)].copy()

    if dhis_filtered.empty:
        return pd.DataFrame()

    # Merge with population
    merged = dhis_filtered.merge(
        pop_df,
        on=unit_cols + ["year"],
        how="left",
        suffixes=("", "_pop"),
    )

    # Drop rows without population data
    merged = merged[merged["population"].notna()]

    if merged.empty:
        return pd.DataFrame()

    # Map indicators to age groups
    merged["age_group"] = merged["indicator"].map(INDICATOR_TO_FRACTION)
    merged = merged[merged["age_group"].notna()]

    # Merge with fractions
    merged = merged.merge(
        fractions_df[["year", "age_group", "fraction"]],
        on=["year", "age_group"],
        how="left",
    )

    # Calculate coverage
    merged["denominator_population"] = merged["population"] * merged["fraction"]
    merged["coverage"] = merged["value"] / merged["denominator_population"]

    # Clip coverage to 0-1
    merged["coverage"] = merged["coverage"].clip(0, 1)

    result_cols = unit_cols + [
        "year",
        "indicator",
        "value",
        "population",
        "age_group",
        "fraction",
        "denominator_population",
        "coverage",
    ]

    return merged[result_cols].drop_duplicates()


def process_regional_level() -> pd.DataFrame:
    """Process regional-level coverage estimates."""
    print("\nProcessing regional level...")

    # Load GPS-based estimates
    dhis_path = Path("estimates/dhis2_yearly_regional_all_indicators_gps.csv")
    if not dhis_path.exists():
        print(f"  Skipping: {dhis_path} not found")
        return pd.DataFrame()

    dhis_df = pd.read_csv(dhis_path)
    # Normalize region names to match population data
    dhis_df["region"] = (
        dhis_df["region"].map(GEO_TO_NORMALIZED_REGION).fillna(dhis_df["region"])
    )
    dhis_df = dhis_df[dhis_df["region"].notna()]

    pop_regional = get_regional_population(pop_df)
    unit_cols = ["region"]

    # Compute population-based coverage
    pop_cov = compute_population_coverage(dhis_df, pop_regional, unit_cols)

    # Compute special indicators
    lbw_cov = compute_lbw_coverage(dhis_df, unit_cols)
    csec_cov = compute_csec_coverage(dhis_df, unit_cols)

    # Combine
    result = pd.concat([pop_cov, lbw_cov, csec_cov], ignore_index=True)
    result = result.sort_values(["indicator", "region", "year"])

    # Compute National aggregates
    national = result.groupby(
        ["year", "indicator", "age_group", "fraction"], as_index=False
    )[["value", "population", "denominator_population"]].sum()
    national["region"] = "National"
    national["coverage"] = national["value"] / national["denominator_population"]
    national = national[result.columns]

    result = pd.concat([result, national], ignore_index=True)
    result = result.sort_values(["indicator", "region", "year"])

    return result


def process_zonal_level(regional_result: pd.DataFrame) -> pd.DataFrame:
    """Process zonal-level coverage estimates with scaling to match regional totals."""
    print("\nProcessing zonal level...")

    dhis_path = Path("estimates/dhis2_yearly_zonal_all_indicators_gps.csv")
    if not dhis_path.exists():
        print(f"  Skipping: {dhis_path} not found")
        return pd.DataFrame()

    dhis_df = pd.read_csv(dhis_path)
    # The zone column from GPS-based estimates is already GeoJSON zone name
    # Normalize region
    dhis_df["region"] = (
        dhis_df["region"].map(GEO_TO_NORMALIZED_REGION).fillna(dhis_df["region"])
    )

    # Get zonal population
    pop_zonal = get_zonal_population(pop_df)

    unit_cols = ["region", "zone"]

    # Compute population-based coverage
    pop_cov = compute_population_coverage(dhis_df, pop_zonal, unit_cols)

    # Compute special indicators
    lbw_cov = compute_lbw_coverage(dhis_df, unit_cols)
    csec_cov = compute_csec_coverage(dhis_df, unit_cols)

    # Combine
    parts = [df for df in [pop_cov, lbw_cov, csec_cov] if not df.empty]
    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)

    assert not regional_result.empty and not result.empty
    # Group zonal results to get implied regional sums
    # We need both sums to calculate aggregated coverage of the reported units
    zonal_stats = (
        result.groupby(["region", "year", "indicator"])
        .agg(
            zonal_sum_value=("value", "sum"),
            zonal_sum_pop=("denominator_population", "sum"),
        )
        .reset_index()
    )

    # Calculate Aggregated Coverage of the Zonal Data
    zonal_stats["agg_zonal_coverage"] = (
        zonal_stats["zonal_sum_value"] / zonal_stats["zonal_sum_pop"]
    )

    target_regions = regional_result[["region", "year", "indicator", "coverage"]].copy()
    target_regions = target_regions.rename(
        columns={"coverage": "target_regional_coverage"}
    )

    # Calculate scaling factors
    merged = zonal_stats.merge(
        target_regions, on=["region", "year", "indicator"], how="inner"
    )

    # Calculate Ratio: Target / Current
    # Guard against zero division
    merged["scale_factor"] = 1.0
    mask_nonzero = merged["agg_zonal_coverage"] > 0
    merged.loc[mask_nonzero, "scale_factor"] = (
        merged.loc[mask_nonzero, "target_regional_coverage"]
        / merged.loc[mask_nonzero, "agg_zonal_coverage"]
    )

    # Merge scaling factor back to result
    result = result.merge(
        merged[["region", "year", "indicator", "scale_factor"]],
        on=["region", "year", "indicator"],
        how="left",
    )

    result["value"] = result["value"] * result["scale_factor"]
    result["coverage"] = result["coverage"] * result["scale_factor"]

    result["coverage"] = result["coverage"].clip(0, 1)

    result = result.sort_values(["indicator", "region", "zone", "year"])

    return result


def process_woreda_level(regional_result: pd.DataFrame) -> pd.DataFrame:
    """Process woreda-level coverage estimates with scaling to match regional totals."""
    print("\nProcessing woreda level...")

    dhis_path = Path("estimates/dhis2_yearly_woreda_all_indicators_gps.csv")

    dhis_df = pd.read_csv(dhis_path)
    # The woreda column from GPS-based estimates is already GeoJSON woreda name
    # Normalize region
    dhis_df["region"] = (
        dhis_df["region"].map(GEO_TO_NORMALIZED_REGION).fillna(dhis_df["region"])
    )

    # Get woreda population (merge on region + woreda only, zone names may differ)
    pop_woreda = get_woreda_population(pop_df)
    pop_woreda_no_zone = pop_woreda.drop(columns=["zone"], errors="ignore")

    unit_cols = ["region", "woreda"]

    # Compute population-based coverage
    dhis_df_for_merge = dhis_df.drop(columns=["zone"], errors="ignore")
    pop_cov = compute_population_coverage(
        dhis_df_for_merge, pop_woreda_no_zone, unit_cols
    )

    # Add zone back
    if not pop_cov.empty:
        zone_lookup = dhis_df[["region", "woreda", "zone"]].drop_duplicates()
        pop_cov = pop_cov.merge(zone_lookup, on=["region", "woreda"], how="left")
        # Reorder columns
        pop_cov = pop_cov[
            [
                "region",
                "zone",
                "woreda",
                "year",
                "indicator",
                "value",
                "population",
                "age_group",
                "fraction",
                "denominator_population",
                "coverage",
            ]
        ]

    # Compute special indicators
    lbw_cov = compute_lbw_coverage(dhis_df, ["region", "zone", "woreda"])
    csec_cov = compute_csec_coverage(dhis_df, ["region", "zone", "woreda"])

    # Combine
    parts = [df for df in [pop_cov, lbw_cov, csec_cov] if not df.empty]
    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)

    assert not regional_result.empty and not result.empty
    # Group woreda results to get implied regional sums
    woreda_stats = (
        result.groupby(["region", "year", "indicator"])
        .agg(
            woreda_sum_value=("value", "sum"),
            woreda_sum_pop=("denominator_population", "sum"),
        )
        .reset_index()
    )

    # Calculate Aggregated Coverage
    woreda_stats["agg_woreda_coverage"] = (
        woreda_stats["woreda_sum_value"] / woreda_stats["woreda_sum_pop"]
    )

    # Merge with independent regional results
    target_regions = regional_result[["region", "year", "indicator", "coverage"]].copy()
    target_regions = target_regions.rename(
        columns={"coverage": "target_regional_coverage"}
    )

    # Calculate scaling factors
    merged = woreda_stats.merge(
        target_regions, on=["region", "year", "indicator"], how="inner"
    )

    merged["scale_factor"] = 1.0
    mask_nonzero = merged["agg_woreda_coverage"] > 0
    merged.loc[mask_nonzero, "scale_factor"] = (
        merged.loc[mask_nonzero, "target_regional_coverage"]
        / merged.loc[mask_nonzero, "agg_woreda_coverage"]
    )

    # Merge scaling factor back
    result = result.merge(
        merged[["region", "year", "indicator", "scale_factor"]],
        on=["region", "year", "indicator"],
        how="left",
    )

    print(
        f"  Applying scaling to woreda estimates (Mean factor: {result['scale_factor'].mean():.4f})"
    )

    result["value"] = result["value"] * result["scale_factor"]
    result["coverage"] = result["coverage"] * result["scale_factor"]

    # Valid range clipping (post-scaling)
    result["coverage"] = result["coverage"].clip(0, 1)

    result = result.sort_values(["indicator", "region", "zone", "woreda", "year"])

    return result


def process_kebele_level(regional_result: pd.DataFrame) -> pd.DataFrame:
    """Process kebele-level coverage estimates with scaling to match regional totals."""
    print("\nProcessing kebele level...")

    dhis_path = Path("estimates/dhis2_yearly_kebele_all_indicators_gps.csv")
    if not dhis_path.exists():
        print(f"  Skipping: {dhis_path} not found")
        return pd.DataFrame()

    dhis_df = pd.read_csv(dhis_path)
    # The kebele column from GPS-based estimates is already GeoJSON kebele name (RK_NAME)
    # Normalize region
    dhis_df["region"] = (
        dhis_df["region"].map(GEO_TO_NORMALIZED_REGION).fillna(dhis_df["region"])
    )

    # Get kebele population
    pop_kebele = get_kebele_population(pop_df)

    unit_cols = ["region", "zone", "woreda", "kebele"]

    # Compute population-based coverage
    pop_cov = compute_population_coverage(dhis_df, pop_kebele, unit_cols)

    # Compute special indicators (LBW, C-Section) - these don't need external population
    lbw_cov = compute_lbw_coverage(dhis_df, unit_cols)
    csec_cov = compute_csec_coverage(dhis_df, unit_cols)

    # Combine
    parts = [df for df in [pop_cov, lbw_cov, csec_cov] if not df.empty]

    result = pd.concat(parts, ignore_index=True)

    assert not regional_result.empty
    # Group kebele results to get implied regional sums
    kebele_stats = (
        result.groupby(["region", "year", "indicator"])
        .agg(
            kebele_sum_value=("value", "sum"),
            kebele_sum_pop=("denominator_population", "sum"),
        )
        .reset_index()
    )

    # Calculate Aggregated Coverage
    kebele_stats["agg_kebele_coverage"] = (
        kebele_stats["kebele_sum_value"] / kebele_stats["kebele_sum_pop"]
    )

    # Merge with independent regional results
    target_regions = regional_result[["region", "year", "indicator", "coverage"]].copy()
    target_regions = target_regions.rename(
        columns={"coverage": "target_regional_coverage"}
    )

    # Calculate scaling factors
    merged = kebele_stats.merge(
        target_regions, on=["region", "year", "indicator"], how="inner"
    )

    merged["scale_factor"] = 1.0
    mask_nonzero = merged["agg_kebele_coverage"] > 0
    merged.loc[mask_nonzero, "scale_factor"] = (
        merged.loc[mask_nonzero, "target_regional_coverage"]
        / merged.loc[mask_nonzero, "agg_kebele_coverage"]
    )

    # Merge scaling factor back
    result = result.merge(
        merged[["region", "year", "indicator", "scale_factor"]],
        on=["region", "year", "indicator"],
        how="left",
    )

    print(
        f"  Applying scaling to kebele estimates (Mean factor: {result['scale_factor'].mean():.4f})"
    )

    result["value"] = result["value"] * result["scale_factor"]
    result["coverage"] = result["coverage"] * result["scale_factor"]

    # Valid range clipping (post-scaling)
    result["coverage"] = result["coverage"].clip(0, 1)

    result = result.sort_values(
        ["indicator", "region", "zone", "woreda", "kebele", "year"]
    )

    return result


# =========================================
# MAIN PROCESSING
# =========================================
if __name__ == "__main__":
    print()

    regional_result = process_regional_level()
    if not regional_result.empty:
        out_path = Path("estimates/dhis2_coverage_estimates_gps.csv")
        regional_result.to_csv(out_path, index=False)
        print(f"Wrote {len(regional_result):,} rows to {out_path}")
        print(
            f"  {regional_result['indicator'].nunique()} indicators, "
            f"{regional_result['region'].nunique()} regions"
        )

    # Zonal level
    zonal_result = process_zonal_level(regional_result)
    if not zonal_result.empty:
        out_path_zonal = Path("estimates/dhis2_coverage_estimates_zonal_gps.csv")
        zonal_result.to_csv(out_path_zonal, index=False)
        print(f"Wrote {len(zonal_result):,} rows to {out_path_zonal}")
        print(
            f"  {zonal_result['indicator'].nunique()} indicators, "
            f"{zonal_result['zone'].nunique()} zones"
        )

    # Woreda level
    woreda_result = process_woreda_level(regional_result)
    if not woreda_result.empty:
        out_path_woreda = Path("estimates/dhis2_coverage_estimates_woreda_gps.csv")
        woreda_result.to_csv(out_path_woreda, index=False)
        print(f"Wrote {len(woreda_result):,} rows to {out_path_woreda}")
        print(
            f"  {woreda_result['indicator'].nunique()} indicators, "
            f"{woreda_result['woreda'].nunique()} woredas"
        )

    # Kebele level
    kebele_result = process_kebele_level(regional_result)
    if not kebele_result.empty:
        out_path_kebele = Path("estimates/dhis2_coverage_estimates_kebele_gps.csv")
        kebele_result.to_csv(out_path_kebele, index=False)
        print(f"Wrote {len(kebele_result):,} rows to {out_path_kebele}")
        print(
            f"  {kebele_result['indicator'].nunique()} indicators, "
            f"{kebele_result['kebele'].nunique()} kebeles"
        )

    if not regional_result.empty:
        print(f"\nYears: {sorted(regional_result['year'].unique())}")
