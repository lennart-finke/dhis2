"""
Compute yearly DHIS2 indicator estimates using GPS-based boundary matching.

This script:
1. Reads raw DHIS2 facility-level data
2. Joins with facility_boundary_mapping.csv (which uses GPS for boundary assignment)
3. Aggregates by GPS-derived boundaries (boundary_region, boundary_zone, boundary_woreda)
4. Produces estimates at regional, zonal, and woreda levels

This provides an alternative to name-based matching that uses actual GPS coordinates
to assign facilities to administrative boundaries where available.

Saves to:
  - estimates/dhis2_yearly_regional_all_indicators_gps.csv
  - estimates/dhis2_yearly_zonal_all_indicators_gps.csv
  - estimates/dhis2_yearly_woreda_all_indicators_gps.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as stats

from dhis2_utils import (
    prepare_dhis2,
    infer_orgunit_level_cols,
    GEO_TO_NORMALIZED_REGION,
    load_woreda_mapping,
    load_zone_mapping,
)


def impute_subregional_gamma(df, regional_df, group_cols, value_col="value"):
    """
    Impute missing years for sub-regional units using a Gamma distribution approach,
    matching the regional imputed sums.
    """
    indicators = df["indicator"].unique()
    years = sorted(regional_df["year"].unique())

    imputed_rows = []

    for indicator in indicators:
        df_ind = df[df["indicator"] == indicator]
        reg_ind = regional_df[regional_df["indicator"] == indicator]

        units = df_ind[group_cols].drop_duplicates()
        if units.empty:
            continue

        units = units.copy()
        units["_key"] = 1
        years_df = pd.DataFrame({"year": years, "_key": 1})
        grid = units.merge(years_df, on="_key").drop("_key", axis=1)
        grid["indicator"] = indicator

        merged = grid.merge(df_ind, on=group_cols + ["year", "indicator"], how="left")
        reg_sums = reg_ind[["region", "year", "value"]].rename(
            columns={"value": "reg_sum"}
        )
        merged = merged.merge(reg_sums, on=["region", "year"], how="left")

        # Fill missing values with 0 initially to represent no data,
        # but we will overwrite them during imputation.
        merged[value_col] = merged[value_col].fillna(0)

        for region in merged["region"].unique():
            reg_mask = merged["region"] == region

            for y_idx in range(len(years) - 1, -1, -1):
                year = years[y_idx]
                mask = reg_mask & (merged["year"] == year)

                # Check if this year needs imputation.
                # We impute if it's 2016, 2017, 2018, or if region is Tigray and year is 2021, 2022, 2023, 2024.
                needs_imputation = False
                if year < 2019:
                    needs_imputation = True
                elif region == "Tigray" and year in [2021, 2022, 2023, 2024]:
                    needs_imputation = True

                if needs_imputation:
                    ref_year = year + 1
                    if ref_year not in years:
                        continue
                    ref_mask = reg_mask & (merged["year"] == ref_year)
                    ref_data = merged.loc[ref_mask, value_col]

                    target_sum = (
                        merged.loc[mask, "reg_sum"].iloc[0] if mask.sum() > 0 else 0
                    )

                    if pd.isna(target_sum) or target_sum == 0 or len(ref_data) == 0:
                        merged.loc[mask, value_col] = 0
                        continue

                    N = len(ref_data)
                    mu = target_sum / N

                    bar_mu = ref_data.mean()
                    bar_var = ref_data.var(ddof=0)

                    if bar_var == 0 or bar_mu == 0:
                        merged.loc[mask, value_col] = mu
                        continue

                    scale_factor = np.sqrt(mu / bar_mu)
                    adjusted_var = bar_var * scale_factor

                    alpha = (mu**2) / adjusted_var
                    nu = adjusted_var / mu

                    ranks = ref_data.rank(method="average")
                    quantiles = (ranks - 0.5) / N

                    imputed = stats.gamma.ppf(quantiles, a=alpha, scale=nu)

                    imputed_sum = np.nansum(imputed)
                    if imputed_sum > 0:
                        imputed = imputed * (target_sum / imputed_sum)

                    merged.loc[mask, value_col] = imputed

        imputed_rows.append(merged)

    if not imputed_rows:
        return df
    res = pd.concat(imputed_rows, ignore_index=True)
    # Ensure no NaNs from Gamma ppf
    res[value_col] = res[value_col].fillna(0)
    return res[group_cols + ["year", "indicator", value_col]]


# Indicators to include
INDICATORS = [
    "EPI_<1 Year Received 3rd Dose Penta",
    "EPI_Children <1 Year Measels- 1st Dose",
    "NUT_Children 6-59 Months received Vitamin A by Age",
    "NUT_Children 6-59 Months Received Vitamin A by Dose",
    "NUT_Children 24 - 59 Months Dewormed",
    "Number of pregnant women who received ANC first visit by gestational week",
    "MAT_Skilled Birth Attendance",
    "MAT_Births attended by Level IV HEW And Nurses at HP",
    "NUT_Children < 5 years Screened for Acute Malnutrition",
    "NUT_Children <5 Years Screened with MAM",
    "NUT_Children <5 Years Screened with SAM",
    "NUT_Live Births Weighed",
    "NUT_Newborns < 2500 gm",
    "MAT_Births By Caesarean Section",
]

BASE_DIR = Path("data/dhis2/25_12_12")
FACILITY_MAPPING = Path("estimates/facility_boundary_mapping.csv")

OUT_PATH_REGIONAL = Path("estimates/dhis2_yearly_regional_all_indicators_gps.csv")
OUT_PATH_ZONAL = Path("estimates/dhis2_yearly_zonal_all_indicators_gps.csv")
OUT_PATH_WOREDA = Path("estimates/dhis2_yearly_woreda_all_indicators_gps.csv")
OUT_PATH_KEBELE = Path("estimates/dhis2_yearly_kebele_all_indicators_gps.csv")

METADATA_COLS = {
    "organisationunitid",
    "organisationunitname",
    "organisationunitcode",
    "organisationunitdescription",
    "periodid",
    "periodname",
    "perioddescription",
    "periodcode",
    "gregorian_date",
    "year",
}


def get_quarter_ordinal(year, quarter):
    """Convert year and quarter to a continuous ordinal (Q1 2016 = 0)."""
    return (year - 2016) * 4 + (quarter - 1)


# Load facility boundary mapping
print("Loading facility boundary mapping for GPS-based matching...")
facility_map = pd.read_csv(FACILITY_MAPPING)
print(f"  Loaded {len(facility_map)} facility mappings")
print(
    f"  GPS-matched: {(facility_map['match_method'] == 'gps').sum()}, "
    f"Name-matched: {(facility_map['match_method'] == 'name').sum()}"
)

# Create lookup dict: organisationunitid -> {boundary_region, boundary_zone, boundary_woreda}
# Drop duplicates by dhis2_id (keep first)
facility_lookup_df = facility_map.drop_duplicates(subset="dhis2_id").set_index(
    "dhis2_id"
)
facility_lookup = facility_lookup_df[
    [
        "boundary_region",
        "boundary_zone",
        "boundary_woreda",
        "boundary_kebele",
        "boundary_region_admin4",
        "boundary_zone_admin4",
        "boundary_woreda_admin4",
    ]
].to_dict("index")

# Load standard DHIS2->GeoJSON name maps generated by match_woreda_names.py
dhis2_woreda_df = load_woreda_mapping()
dhis2_woreda_to_geojson = (
    dhis2_woreda_df.dropna(subset=["woreda_dhis2", "woreda_geojson"])
    .set_index("woreda_dhis2")["woreda_geojson"]
    .to_dict()
)

dhis2_zone_df = load_zone_mapping()
dhis2_zone_to_geojson = (
    dhis2_zone_df.dropna(subset=["zone_dhis2", "zone_geojson"])
    .set_index("zone_dhis2")["zone_geojson"]
    .to_dict()
)

all_data = []

print("\nLoading DHIS2 data...")
for csv_path in sorted(BASE_DIR.glob("*.csv")):
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=True)

    df = prepare_dhis2(df, csv_path=csv_path)

    # Filter columns to only those in INDICATORS
    level_cols = set(infer_orgunit_level_cols(df.columns))
    exclude = METADATA_COLS | level_cols
    available_indicators = [c for c in df.columns if c in INDICATORS]

    assert available_indicators

    # Keep organisationunitid for mapping
    level_cols_list = list(infer_orgunit_level_cols(df.columns))
    keep_cols = (
        ["organisationunitid", "gregorian_date"]
        + available_indicators
        + level_cols_list
    )
    df_subset = df[keep_cols].copy()

    for col in available_indicators:
        df_subset[col] = pd.to_numeric(df_subset[col], errors="coerce").fillna(0.0)

    # Map facilities to GPS-based boundaries
    df_subset["boundary_region"] = df_subset["organisationunitid"].map(
        lambda x: facility_lookup.get(x, {}).get("boundary_region")
    )
    df_subset["boundary_zone"] = df_subset["organisationunitid"].map(
        lambda x: facility_lookup.get(x, {}).get("boundary_zone")
    )
    df_subset["boundary_woreda"] = df_subset["organisationunitid"].map(
        lambda x: facility_lookup.get(x, {}).get("boundary_woreda")
    )
    df_subset["boundary_kebele"] = df_subset["organisationunitid"].map(
        lambda x: facility_lookup.get(x, {}).get("boundary_kebele")
    )
    df_subset["boundary_region_admin4"] = df_subset["organisationunitid"].map(
        lambda x: facility_lookup.get(x, {}).get("boundary_region_admin4")
    )
    df_subset["boundary_zone_admin4"] = df_subset["organisationunitid"].map(
        lambda x: facility_lookup.get(x, {}).get("boundary_zone_admin4")
    )
    df_subset["boundary_woreda_admin4"] = df_subset["organisationunitid"].map(
        lambda x: facility_lookup.get(x, {}).get("boundary_woreda_admin4")
    )

    # Filter to facilities that couldn't be mapped
    missing_mask = df_subset["boundary_region"].isna()
    if missing_mask.any() and "orgunitlevel2" in df_subset.columns:
        # Map Region
        df_subset.loc[missing_mask, "boundary_region"] = (
            df_subset.loc[missing_mask, "orgunitlevel2"]
            .map(GEO_TO_NORMALIZED_REGION)
            .fillna(df_subset.loc[missing_mask, "orgunitlevel2"])
        )

        # Determine Zone column (usually level 3)
        if "orgunitlevel3" in df_subset.columns:
            df_subset.loc[missing_mask, "boundary_zone"] = (
                df_subset.loc[missing_mask, "orgunitlevel3"]
                .map(dhis2_zone_to_geojson)
                .fillna(df_subset.loc[missing_mask, "orgunitlevel3"])
            )

        # Determine Woreda column (usually level 4)
        if "orgunitlevel4" in df_subset.columns:
            df_subset.loc[missing_mask, "boundary_woreda"] = (
                df_subset.loc[missing_mask, "orgunitlevel4"]
                .map(dhis2_woreda_to_geojson)
                .fillna(df_subset.loc[missing_mask, "orgunitlevel4"])
            )

        if "boundary_zone" in df_subset.columns:
            unmapped_zone_mask = missing_mask & ~df_subset["orgunitlevel3"].isin(
                dhis2_zone_to_geojson.keys()
            )
            if unmapped_zone_mask.any():
                df_subset.loc[unmapped_zone_mask, "boundary_zone"] = (
                    df_subset.loc[unmapped_zone_mask, "boundary_zone"]
                    .str.replace(r"(?i)\s+zone$", "", regex=True)
                    .str.strip()
                )

    # Normalize boundary_woreda names to match GeoJSON for ALL rows.
    # 1) Apply DHIS2→GeoJSON name mapping first (keys include suffixes like "Woreda")
    # 2) Then strip remaining "Woreda"/"Town" suffixes as fallback
    if "boundary_woreda" in df_subset.columns:
        df_subset["boundary_woreda"] = (
            df_subset["boundary_woreda"]
            .map(dhis2_woreda_to_geojson)
            .fillna(df_subset["boundary_woreda"])
        )
        df_subset["boundary_woreda"] = (
            df_subset["boundary_woreda"]
            .str.replace(r"(?i)\s+woreda$", "", regex=True)
            .str.replace(r"(?i)\s+town$", "", regex=True)
            .str.strip()
        )

    # Normalize boundary_region for ALL rows to consolidate variants
    # (e.g. "Oromia Region" → "Oromia", "Gambella Region" → "Gambela")
    if "boundary_region" in df_subset.columns:
        df_subset["boundary_region"] = (
            df_subset["boundary_region"]
            .map(GEO_TO_NORMALIZED_REGION)
            .fillna(df_subset["boundary_region"])
        )

    # Melt to long format
    id_vars = [
        "organisationunitid",
        "gregorian_date",
        "boundary_region",
        "boundary_zone",
        "boundary_woreda",
        "boundary_kebele",
        "boundary_region_admin4",
        "boundary_zone_admin4",
        "boundary_woreda_admin4",
    ]
    melted = df_subset.melt(
        id_vars=id_vars,
        value_vars=available_indicators,
        var_name="indicator",
        value_name="value",
    )

    all_data.append(melted)

full_df = pd.concat(all_data, ignore_index=True)

# Add time info
full_df["year"] = full_df["gregorian_date"].dt.year
full_df["quarter"] = (full_df["gregorian_date"].dt.month - 1) // 3 + 1

# Get max year from data
max_year = full_df["year"].max()

# Generate full grid of quarters
years = range(2016, max_year + 1)
quarters = range(1, 5)
all_quarters = [(y, q) for y in years for q in quarters]

# =========================================
# REGIONAL LEVEL (using GPS-based boundaries)
# =========================================
print("\nAggregating to regional level (GPS-based)...")
regional_quarterly = full_df.groupby(
    ["boundary_region", "indicator", "year", "quarter"], as_index=False
)["value"].sum()

print("Computing regional estimates...")
regional_results = []
regions = regional_quarterly["boundary_region"].unique()
indicators = regional_quarterly["indicator"].unique()

# ANC was discontinued after Q3 2022 (national totals drop from ~1.3M/quarter to near-zero
# starting Q1 2023). Cap at 2022 so we use real observed data through its last valid year.
ANC_INDICATOR = (
    "Number of pregnant women who received ANC first visit by gestational week"
)
ANC_MAX_YEAR = 2022

for region in regions:
    for indicator in indicators:
        subset = regional_quarterly[
            (regional_quarterly["boundary_region"] == region)
            & (regional_quarterly["indicator"] == indicator)
        ].copy()

        if subset.empty:
            continue

        subset["ordinal"] = subset.apply(
            lambda r: get_quarter_ordinal(r["year"], r["quarter"]), axis=1
        )

        min_ordinal_for_fit = get_quarter_ordinal(2019, 1)
        fit_mask = subset["ordinal"] >= min_ordinal_for_fit

        # Exclude Tigray from conflict period
        if region == "Tigray":
            idx_start = get_quarter_ordinal(2021, 1)
            idx_end = get_quarter_ordinal(2024, 2)
            fit_mask = fit_mask & ~(
                (subset["ordinal"] >= idx_start) & (subset["ordinal"] <= idx_end)
            )

        data_for_fit = subset[fit_mask]

        assert len(data_for_fit) >= 2

        X = data_for_fit["ordinal"].values
        y = data_for_fit["value"].values
        slope, intercept = np.polyfit(X, y, 1)

        observed_quarters = set(
            subset[["year", "quarter"]].itertuples(index=False, name=None)
        )

        if indicator == ANC_INDICATOR:
            quarters_to_fill = [(y, q) for y, q in all_quarters if y <= ANC_MAX_YEAR]
        else:
            quarters_to_fill = all_quarters

        quarterly_values = []
        for y_val, q in quarters_to_fill:
            if (y_val, q) in observed_quarters:
                obs_value = subset[
                    (subset["year"] == y_val) & (subset["quarter"] == q)
                ]["value"].iloc[0]
                quarterly_values.append(
                    {"year": y_val, "quarter": q, "value": obs_value}
                )
            else:
                ordinal = get_quarter_ordinal(y_val, q)
                pred_value = max(0, slope * ordinal + intercept)
                quarterly_values.append(
                    {"year": y_val, "quarter": q, "value": pred_value}
                )

        pred_df = pd.DataFrame(quarterly_values)
        yearly_sums = pred_df.groupby("year")["value"].sum().reset_index()

        for _, row in yearly_sums.iterrows():
            regional_results.append(
                {
                    "region": region,
                    "year": int(row["year"]),
                    "indicator": indicator,
                    "value": row["value"],
                }
            )

regional_df = pd.DataFrame(regional_results)

# Add national aggregates
national = regional_df.groupby(["year", "indicator"], as_index=False)["value"].sum()
national["region"] = "National"
regional_df = pd.concat([regional_df, national], ignore_index=True)
regional_df = regional_df.sort_values(["indicator", "region", "year"])

OUT_PATH_REGIONAL.parent.mkdir(parents=True, exist_ok=True)
regional_df.to_csv(OUT_PATH_REGIONAL, index=False)
print(f"Wrote {len(regional_df):,} rows to {OUT_PATH_REGIONAL}")
print(
    f"  {regional_df['indicator'].nunique()} indicators, {regional_df['region'].nunique()} regions"
)


def redistribute_dummy_counts(df, group_cols, value_col, dummy_col, dummy_val):
    """
    Distributes values from rows where dummy_col == dummy_val to other rows
    within the same group (defined by group_cols).
    """
    is_dummy = df[dummy_col] == dummy_val
    if not is_dummy.any():
        return df

    # Sum of dummy values per group
    dummy_sums = df[is_dummy].groupby(group_cols)[value_col].sum().reset_index()
    dummy_sums = dummy_sums.rename(columns={value_col: "dummy_sum"})

    # Non-dummy rows
    non_dummy = df[~is_dummy].copy()

    # Sum of non-dummy values per group
    non_dummy_sums = non_dummy.groupby(group_cols)[value_col].sum().reset_index()
    non_dummy_sums = non_dummy_sums.rename(columns={value_col: "valid_sum"})

    # Merge sums (outer join to capture groups that might only have dummy data)
    sums = pd.merge(non_dummy_sums, dummy_sums, on=group_cols, how="outer").fillna(0)

    # Calculate factor: 1 + (dummy_sum / valid_sum)
    # If valid_sum is 0, factor is 1 (we can't distribute)
    sums["factor"] = 1.0
    valid_mask = sums["valid_sum"] > 0
    sums.loc[valid_mask, "factor"] = 1.0 + (
        sums.loc[valid_mask, "dummy_sum"] / sums.loc[valid_mask, "valid_sum"]
    )

    # Apply factor to non-dummy rows
    merged = pd.merge(
        non_dummy, sums[group_cols + ["factor"]], on=group_cols, how="left"
    )
    merged["factor"] = merged["factor"].fillna(1.0)
    merged[value_col] = merged[value_col] * merged["factor"]

    # Identify dummy rows to keep (where valid_sum == 0 but dummy_sum > 0)
    # We keep these because we couldn't distribute them
    keep_groups = sums[(sums["valid_sum"] == 0) & (sums["dummy_sum"] > 0)][group_cols]

    if not keep_groups.empty:
        dummy_rows_to_keep = df[is_dummy].merge(keep_groups, on=group_cols, how="inner")
        final_df = pd.concat(
            [merged.drop(columns=["factor"]), dummy_rows_to_keep], ignore_index=True
        )
    else:
        final_df = merged.drop(columns=["factor"])

    return final_df


# ZONAL LEVEL (using GPS-based boundaries)
print("\nAggregating to zonal level (GPS-based)...")
zonal_quarterly = full_df.groupby(
    ["boundary_region", "boundary_zone", "indicator", "year", "quarter"], as_index=False
)["value"].sum()

# Process zonal level with simple yearly aggregation (no imputation for now)
zonal_yearly = zonal_quarterly.groupby(
    ["boundary_region", "boundary_zone", "indicator", "year"], as_index=False
)["value"].sum()

zonal_yearly = zonal_yearly.rename(
    columns={"boundary_region": "region", "boundary_zone": "zone"}
)

# Distribute "Unidentified Zone" counts proportionally within each Region
zonal_yearly = redistribute_dummy_counts(
    zonal_yearly,
    group_cols=["region", "year", "indicator"],
    value_col="value",
    dummy_col="zone",
    dummy_val="Unidentified Zone",
)

print("Imputing zonal estimates using Gamma distribution...")
zonal_yearly = impute_subregional_gamma(
    zonal_yearly, regional_df, group_cols=["region", "zone"], value_col="value"
)

zonal_yearly = zonal_yearly[["region", "zone", "year", "indicator", "value"]]
zonal_yearly = zonal_yearly.sort_values(["indicator", "region", "zone", "year"])

OUT_PATH_ZONAL.parent.mkdir(parents=True, exist_ok=True)
zonal_yearly.to_csv(OUT_PATH_ZONAL, index=False)
print(f"Wrote {len(zonal_yearly):,} rows to {OUT_PATH_ZONAL}")
print(
    f"  {zonal_yearly['indicator'].nunique()} indicators, {zonal_yearly['zone'].nunique()} zones"
)

# WOREDA LEVEL (using GPS-based boundaries)
print("\nAggregating to woreda level (GPS-based)...")
woreda_quarterly = full_df.groupby(
    [
        "boundary_region",
        "boundary_zone",
        "boundary_woreda",
        "indicator",
        "year",
        "quarter",
    ],
    as_index=False,
)["value"].sum()

# Process woreda level with simple yearly aggregation
woreda_yearly = woreda_quarterly.groupby(
    ["boundary_region", "boundary_zone", "boundary_woreda", "indicator", "year"],
    as_index=False,
)["value"].sum()

woreda_yearly = woreda_yearly.rename(
    columns={
        "boundary_region": "region",
        "boundary_zone": "zone",
        "boundary_woreda": "woreda",
    }
)

# Step 1: Distribute "Unidentified Woreda" counts proportionally within each Zone
woreda_yearly = redistribute_dummy_counts(
    woreda_yearly,
    group_cols=["region", "zone", "year", "indicator"],
    value_col="value",
    dummy_col="woreda",
    dummy_val="Unidentified Woreda",
)

# Step 2: Distribute "Unidentified Zone" counts proportionally within each Region
# Note: This distributes counts from rows where zone="Unidentified Zone" to other zones.
woreda_yearly = redistribute_dummy_counts(
    woreda_yearly,
    group_cols=["region", "year", "indicator"],
    value_col="value",
    dummy_col="zone",
    dummy_val="Unidentified Zone",
)

print("Imputing woreda estimates using Gamma distribution...")
woreda_yearly = impute_subregional_gamma(
    woreda_yearly,
    regional_df,
    group_cols=["region", "zone", "woreda"],
    value_col="value",
)

woreda_yearly = woreda_yearly[
    ["region", "zone", "woreda", "year", "indicator", "value"]
]
woreda_yearly = woreda_yearly.sort_values(
    ["indicator", "region", "zone", "woreda", "year"]
)

OUT_PATH_WOREDA.parent.mkdir(parents=True, exist_ok=True)
woreda_yearly.to_csv(OUT_PATH_WOREDA, index=False)
print(f"Wrote {len(woreda_yearly):,} rows to {OUT_PATH_WOREDA}")
print(
    f"  {woreda_yearly['indicator'].nunique()} indicators, {woreda_yearly['woreda'].nunique()} woredas"
)

# Placeholder replacement (no-op pending view)
OUT_PATH_KEBELE = Path("estimates/dhis2_yearly_kebele_all_indicators_gps.csv")
print("\nAggregating to kebele level (GPS-based)...")
# Normalize the admin4 region names to match population file (which uses standardized mapping)
full_df["boundary_region_admin4_norm"] = (
    full_df["boundary_region_admin4"]
    .map(GEO_TO_NORMALIZED_REGION)
    .fillna(full_df["boundary_region_admin4"])
)

# Fill NaNs in grouping columns to prevent dropping rows
# Region and Kebele should be present for GPS-matched rows, but Zone/Woreda might be empty in Admin 4 attributes
full_df["boundary_zone_admin4"] = full_df["boundary_zone_admin4"].fillna("")
full_df["boundary_woreda_admin4"] = full_df["boundary_woreda_admin4"].fillna("")

kebele_quarterly = full_df.groupby(
    [
        "boundary_region_admin4_norm",
        "boundary_zone_admin4",
        "boundary_woreda_admin4",
        "boundary_kebele",
        "indicator",
        "year",
        "quarter",
    ],
    as_index=False,
    dropna=False,  # Also explicit dropna=False
)["value"].sum()

# Process kebele level with simple yearly aggregation
kebele_yearly = kebele_quarterly.groupby(
    [
        "boundary_region_admin4_norm",
        "boundary_zone_admin4",
        "boundary_woreda_admin4",
        "boundary_kebele",
        "indicator",
        "year",
    ],
    as_index=False,
    dropna=False,
)["value"].sum()

kebele_yearly = kebele_yearly.rename(
    columns={
        "boundary_region_admin4_norm": "region",
        "boundary_zone_admin4": "zone",
        "boundary_woreda_admin4": "woreda",
        "boundary_kebele": "kebele",
    }
)

# Drop rows where kebele is missing (since we only have it for GPS-matched facilities)
kebele_yearly = kebele_yearly[kebele_yearly["kebele"].notna()]

print("Imputing kebele estimates using Gamma distribution...")
kebele_yearly = impute_subregional_gamma(
    kebele_yearly,
    regional_df,
    group_cols=["region", "zone", "woreda", "kebele"],
    value_col="value",
)

kebele_yearly = kebele_yearly[
    ["region", "zone", "woreda", "kebele", "year", "indicator", "value"]
]
kebele_yearly = kebele_yearly.sort_values(
    ["indicator", "region", "zone", "woreda", "kebele", "year"]
)

OUT_PATH_KEBELE.parent.mkdir(parents=True, exist_ok=True)
kebele_yearly.to_csv(OUT_PATH_KEBELE, index=False)
print(f"Wrote {len(kebele_yearly):,} rows to {OUT_PATH_KEBELE}")
print(
    f"  {kebele_yearly['indicator'].nunique()} indicators, {kebele_yearly['kebele'].nunique()} kebeles"
)

print(f"\nYears: {sorted(regional_df['year'].unique())}")
