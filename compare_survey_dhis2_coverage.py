"""
Compare DHS and PMA coverage estimates with DHIS2 coverage estimates, using beta regression.
Combines data from both sources for calibration where available.
Saves to estimates/dhs_pma_dhis2_coverage_comparison.csv in long format.
Creates scatterplots for each indicator comparison.

ADMIN_LEVEL is the administrative level of the dependent variable, the survey coverage.
ADMIN_LEVEL_INDEP is the administrative level of the independent variable, the DHIS2 coverage.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.integrate import quad
from scipy.special import expit
import stats_utils

# Change this to control the level of analysis
ADMIN_LEVEL = "regional"  # Options: "regional", "zonal", "woreda"
ADMIN_LEVEL_INDEP = "woreda"  # Options: "woreda", "zonal", controls level of independent variable for ecological regression

# Ecological regression is applied to all indicators except Low Birth Weight and Deworming.
COMPUTE_COOKS_DISTANCE = True

# Map DHS indicators to DHIS2 indicators
DHS_INDICATOR_MAP = {
    "ANC First Visit": "Number of pregnant women who received ANC first visit by gestational week",
    "Skilled Birth Attendance": "MAT_Skilled Birth Attendance",
    "Measles 1st Dose": "EPI_Children <1 Year Measels- 1st Dose",
    "DPT3/Penta3": "EPI_<1 Year Received 3rd Dose Penta",
    "Drugs for Intestinal Parasites": "NUT_Children 24 - 59 Months Dewormed",
    "Low Birth Weight": "Low Birth Weight",
    "MAM": "NUT_Children <5 Years Screened with MAM",
    "C-section": "MAT_Births By Caesarean Section",
}


# Map PMA indicators to DHIS2 indicators
PMA_INDICATOR_MAP = {
    "0-12 months Pentavalent3": "EPI_<1 Year Received 3rd Dose Penta",
    "0-12 months Measles1": "EPI_Children <1 Year Measels- 1st Dose",
    "0-12 months Vitamin A": "NUT_Children 6-59 Months received Vitamin A by Age",
    "Mother dewormed during pregnancy": "NUT_Children 24 - 59 Months Dewormed",
    "Post-partum had ANC": "Number of pregnant women who received ANC first visit by gestational week",
    "SBA": "MAT_Skilled Birth Attendance",
    "C-section": "MAT_Births By Caesarean Section",
}

if ADMIN_LEVEL == "regional":
    DHS_FILE = "estimates/dhs_regional_estimates.csv"
    PMA_FILE = "estimates/pma_regional_estimates.csv"
    DHIS2_FILE = "estimates/dhis2_coverage_estimates_gps.csv"
    UNIT_COLS = ["region"]
    UNIT_LABEL = "region"
elif ADMIN_LEVEL == "zonal":
    DHS_FILE = "estimates/dhs_zonal_estimates.csv"
    PMA_FILE = "estimates/pma_zonal_estimates.csv"
    DHIS2_FILE = "estimates/dhis2_coverage_estimates_zonal_gps.csv"
    UNIT_COLS = ["region", "zone"]
    UNIT_LABEL = "zone"
elif ADMIN_LEVEL == "woreda":
    DHS_FILE = "estimates/dhs_woreda_estimates.csv"
    PMA_FILE = "estimates/pma_woreda_estimates.csv"
    DHIS2_FILE = "estimates/dhis2_coverage_estimates_woreda_gps.csv"
    UNIT_COLS = ["region", "zone", "woreda"]
    UNIT_LABEL = "woreda"
else:
    raise ValueError(
        f"Invalid ADMIN_LEVEL: {ADMIN_LEVEL}. Use 'regional', 'zonal', or 'woreda'"
    )

print(f"Running comparison at {ADMIN_LEVEL} level")
if ADMIN_LEVEL == "regional":
    print(
        "Using ECOLOGICAL BETA REGRESSION (woreda-level DHIS2 → regional survey) for applicable indicators"
    )

dhs = pd.read_csv(DHS_FILE)
pma = pd.read_csv(PMA_FILE)
dhis2 = pd.read_csv(DHIS2_FILE)

# Load indep-level DHIS2 data for ecological regression (regional level only)
# Note: region harmonization is applied after region_map is defined below
dhis2_indep = None
if ADMIN_LEVEL == "regional":
    if ADMIN_LEVEL_INDEP == "zonal":
        DHIS2_INDEP_FILE = "estimates/dhis2_coverage_estimates_zonal_gps.csv"
    else:
        DHIS2_INDEP_FILE = "estimates/dhis2_coverage_estimates_woreda_gps.csv"
    dhis2_indep = pd.read_csv(DHIS2_INDEP_FILE)

# Load metaanalysis data for LBW (regional level only)
if ADMIN_LEVEL == "regional":
    meta = pd.read_csv("data/papers/habtegiorgis2023/metaanalysis.csv")
    meta = meta.rename(columns={"prevalence": "coverage", "sample_size": "denominator"})
    meta["indicator"] = "Low Birth Weight"
    meta["source"] = "Metaanalysis"
else:
    meta = pd.DataFrame()

# Harmonize region names
region_map = {
    "Oromiya": "Oromia",
    "Benishangul-Gumuz": "Benishangul Gumuz",
    "Benishangul Gumz": "Benishangul Gumuz",
    "Gambella": "Gambela",
    "Addis": "Addis Ababa",
}
dhs["region"] = dhs["region"].replace(region_map)
pma["region"] = pma["region"].replace(region_map)
dhis2["region"] = dhis2["region"].replace(region_map)
if not meta.empty:
    meta["region"] = meta["region"].replace(region_map)
if dhis2_indep is not None:
    dhis2_indep["region"] = dhis2_indep["region"].replace(region_map)

# Exclude Tigray during conflict years (Q1 2021 - Q2 2024)
# We exclude the full years 2021-2024 to be safe given annual data
tigray_exclude_years = [2021, 2022, 2023, 2024]
dhis2 = dhis2[
    ~((dhis2["region"] == "Tigray") & (dhis2["year"].isin(tigray_exclude_years)))
]

if dhis2_indep is not None:
    dhis2_indep = dhis2_indep[
        ~(
            (dhis2_indep["region"] == "Tigray")
            & (dhis2_indep["year"].isin(tigray_exclude_years))
        )
    ]
# --- Process DHS ---
dhs["dhis2_indicator"] = dhs["indicator"].map(DHS_INDICATOR_MAP)
dhs = dhs.dropna(subset=["coverage", "dhis2_indicator"])
dhs["source"] = "DHS"

# Build columns list for DHS prep
dhs_cols = UNIT_COLS + [
    "year",
    "dhis2_indicator",
    "indicator",
    "coverage",
    "denominator",
    "source",
]
dhs_prep = dhs[[c for c in dhs_cols if c in dhs.columns]].rename(
    columns={
        "coverage": "survey_coverage",
        "indicator": "survey_indicator",
        "denominator": "survey_denominator",
    }
)

# --- Process PMA ---
pma["dhis2_indicator"] = pma["indicator"].map(PMA_INDICATOR_MAP)
pma = pma.dropna(subset=["coverage", "dhis2_indicator"])
pma["source"] = "PMA"

pma_cols = UNIT_COLS + [
    "year",
    "dhis2_indicator",
    "indicator",
    "coverage",
    "denominator",
    "source",
]
pma_prep = pma[[c for c in pma_cols if c in pma.columns]].rename(
    columns={
        "coverage": "survey_coverage",
        "indicator": "survey_indicator",
        "denominator": "survey_denominator",
    }
)

# --- Process Metaanalysis (regional only) ---
if ADMIN_LEVEL == "regional":
    assert not meta.empty
    meta["dhis2_indicator"] = "Low Birth Weight"
    meta_cols = [
        "region",
        "year",
        "dhis2_indicator",
        "indicator",
        "coverage",
        "denominator",
        "source",
    ]
    meta_prep = meta[[c for c in meta_cols if c in meta.columns]].rename(
        columns={
            "coverage": "survey_coverage",
            "indicator": "survey_indicator",
            "denominator": "survey_denominator",
        }
    )
else:
    meta_prep = pd.DataFrame()

# Combine Survey Data
survey_parts = [dhs_prep, pma_prep]
if not meta_prep.empty:
    survey_parts.append(meta_prep)
combined_survey = pd.concat(survey_parts, ignore_index=True)


dhis2_cols = UNIT_COLS + ["year", "indicator", "coverage", "denominator_population"]
dhis2_prep = dhis2[[c for c in dhis2_cols if c in dhis2.columns]].rename(
    columns={
        "coverage": "dhis2_coverage",
        "indicator": "dhis2_indicator",
        "denominator_population": "dhis2_denominator",
    }
)

# Handle SNNP breakup (regional level): aggregate DHIS2 subregions based on year
if ADMIN_LEVEL == "regional":
    dhis2_aggregated_rows = []
    for year in dhis2_prep["year"].unique():
        year_data = dhis2_prep[dhis2_prep["year"] == year].copy()

        # Determine which regions to aggregate based on year
        regions_to_aggregate = ["SNNP"]
        if year <= 2019:
            regions_to_aggregate.extend(["Sidama", "South West Ethiopia"])
        elif year == 2020:
            regions_to_aggregate.append("South West Ethiopia")

        snnp_subregions = year_data[year_data["region"].isin(regions_to_aggregate)]

        if not snnp_subregions.empty:
            snnp_aggregated = (
                snnp_subregions.groupby("dhis2_indicator")
                .apply(
                    lambda g: pd.Series(
                        {
                            "region": "SNNP",
                            "year": year,
                            "dhis2_denominator": g["dhis2_denominator"].sum(),
                            "dhis2_coverage": (
                                (g["dhis2_coverage"] * g["dhis2_denominator"]).sum()
                                / g["dhis2_denominator"].sum()
                                if g["dhis2_denominator"].sum() > 0
                                else 0
                            ),
                        }
                    ),
                    include_groups=False,
                )
                .reset_index()
            )
            dhis2_aggregated_rows.append(snnp_aggregated)

    if dhis2_aggregated_rows:
        dhis2_aggregated = pd.concat(dhis2_aggregated_rows, ignore_index=True)
        # Only remove rows for the specific (region, year) pairs that were aggregated.
        # SNNP is replaced by its aggregated version for all years.
        # Sidama is only aggregated into SNNP for years <= 2019 (became independent in 2020).
        # South West Ethiopia is only aggregated into SNNP for years <= 2020 (became independent in 2021).
        # For later years, these regions exist independently and must be preserved.
        mask_remove = (
            (dhis2_prep["region"] == "SNNP")
            | ((dhis2_prep["region"] == "Sidama") & (dhis2_prep["year"] <= 2019))
            | (
                (dhis2_prep["region"] == "South West Ethiopia")
                & (dhis2_prep["year"] <= 2020)
            )
        )
        dhis2_prep = dhis2_prep[~mask_remove]
        dhis2_prep = pd.concat([dhis2_prep, dhis2_aggregated], ignore_index=True)

# filter out dhis2 coverage == 0, it probably indicates bad matching
dhis2_prep = dhis2_prep[dhis2_prep["dhis2_coverage"] > 0]

merge_keys = UNIT_COLS + ["year", "dhis2_indicator"]
merged = combined_survey.merge(dhis2_prep, on=merge_keys, how="inner")

# Remove National level from regression and analysis
merged = merged[merged["region"] != "National"]

negative_controls = [
    {
        "name": "Negative Control: Measles vs SBA",
        "dhis2_ind": "EPI_Children <1 Year Measels- 1st Dose",
        "survey_ind": "MAT_Skilled Birth Attendance",
    },
    {
        "name": "Negative Control: SBA vs Measles",
        "dhis2_ind": "MAT_Skilled Birth Attendance",
        "survey_ind": "EPI_Children <1 Year Measels- 1st Dose",
    },
    {
        "name": "Negative Control: C-sections vs SBA",
        "dhis2_ind": "MAT_Births By Caesarean Section",
        "survey_ind": "MAT_Skilled Birth Attendance",
    },
]

neg_dfs = []
neg_merge_keys = UNIT_COLS + ["year"]

for nc in negative_controls:
    d_sub = dhis2_prep[dhis2_prep["dhis2_indicator"] == nc["dhis2_ind"]].copy()
    s_sub = combined_survey[
        combined_survey["dhis2_indicator"] == nc["survey_ind"]
    ].copy()

    # Drop indicator col to avoid suffixes/conflicts
    d_sub = d_sub.drop(columns=["dhis2_indicator"])
    s_sub = s_sub.drop(columns=["dhis2_indicator"])

    nc_df = s_sub.merge(d_sub, on=neg_merge_keys, how="inner")
    nc_df["dhis2_indicator"] = nc["name"]
    nc_df = nc_df[nc_df["region"] != "National"]
    neg_dfs.append(nc_df)

if neg_dfs:
    merged = pd.concat([merged] + neg_dfs, ignore_index=True)


merged["difference"] = merged["survey_coverage"] - merged["dhis2_coverage"]
merged["ratio"] = merged["survey_coverage"] / merged["dhis2_coverage"]
merged["abs_difference"] = merged["difference"].abs()

CLIP_THRESHOLD = 0.01

# Clip survey coverage to (0.01, 0.99)
merged["survey_coverage"] = merged["survey_coverage"].clip(
    CLIP_THRESHOLD, 1 - CLIP_THRESHOLD
)

assert (
    max(merged["survey_coverage"]) <= 1
), f"max(merged['survey_coverage']) = {max(merged['survey_coverage'])}"
assert min(merged["survey_coverage"]) >= 0
# Compute combined weight: 1 / (Var(p1) + Var(p2))
survey_var = (
    merged["survey_coverage"].clip(CLIP_THRESHOLD, 1 - CLIP_THRESHOLD)
    * (1 - merged["survey_coverage"].clip(CLIP_THRESHOLD, 1 - CLIP_THRESHOLD))
    / merged["survey_denominator"]
)
dhis2_var = (
    merged["dhis2_coverage"].clip(CLIP_THRESHOLD, 1 - CLIP_THRESHOLD)
    * (1 - merged["dhis2_coverage"].clip(CLIP_THRESHOLD, 1 - CLIP_THRESHOLD))
    / merged["dhis2_denominator"]
)
merged["combined_weight"] = 1 / (survey_var + dhis2_var + 1e-9)
merged["combined_weight"] = merged["combined_weight"].replace(
    [float("inf"), -float("inf")], float("nan")
)
merged["combined_weight"] = merged["combined_weight"].fillna(0).clip(lower=0)

# Identify which DHIS2 indicators are mapped in DHS (for filtering saving)
dhs_mapped_indicators = set(DHS_INDICATOR_MAP.values())

r2_weighted = {}
regression_params = {}
regression_results_cache = {}
fit_metrics = []

# Prepare indep-level DHIS2 data for ecological regression
if ADMIN_LEVEL == "regional" and dhis2_indep is not None:
    # Prepare indep data with required columns
    dhis2_indep_prep = dhis2_indep.rename(
        columns={
            "indicator": "dhis2_indicator",
            "coverage": "indep_coverage",
        }
    )
    # Handle SNNP subregion aggregation for indep data (same logic as regional)
    # Map Sidama and South West Ethiopia woredas/zones to SNNP based on year
    for year in dhis2_indep_prep["year"].unique():
        if year <= 2019:
            mask = (dhis2_indep_prep["year"] == year) & (
                dhis2_indep_prep["region"].isin(["Sidama", "South West Ethiopia"])
            )
            dhis2_indep_prep.loc[mask, "region"] = "SNNP"
        elif year == 2020:
            mask = (dhis2_indep_prep["year"] == year) & (
                dhis2_indep_prep["region"] == "South West Ethiopia"
            )
            dhis2_indep_prep.loc[mask, "region"] = "SNNP"

    # Add Negative Control indep data
    indep_nc_dfs = []
    for nc in negative_controls:
        # Reuse 'negative_controls' list defined earlier
        w_sub = dhis2_indep_prep[
            dhis2_indep_prep["dhis2_indicator"] == nc["dhis2_ind"]
        ].copy()
        if not w_sub.empty:
            w_sub["dhis2_indicator"] = nc["name"]
            indep_nc_dfs.append(w_sub)

    if indep_nc_dfs:
        dhis2_indep_prep = pd.concat(
            [dhis2_indep_prep] + indep_nc_dfs, ignore_index=True
        )

for dhis2_ind, group in merged.groupby("dhis2_indicator"):
    # Check if more than one dataset (source) exists for this indicator
    n_datasets = group["source"].nunique()
    dataset_col = "source" if n_datasets > 1 else None

    print(f"{dhis2_ind} (n={len(group)})")

    # Define region fixed effect (Zonal only)
    region_fe_col = "region" if ADMIN_LEVEL == "zonal" else None

    # Use ecological regression for regional level if enabled
    use_eco = ADMIN_LEVEL == "regional" and dhis2_ind not in [
        "Low Birth Weight",
        "NUT_Children 24 - 59 Months Dewormed",
    ]
    if use_eco:
        # Filter indep data for this indicator
        indep_subset = dhis2_indep_prep[
            dhis2_indep_prep["dhis2_indicator"] == dhis2_ind
        ].copy()

        assert len(indep_subset) > 0, f"No indep data for {dhis2_ind}"
        result = stats_utils.regression_ecological(
            data=group,
            x_col="dhis2_coverage",
            y_col="survey_coverage",
            weight_col="combined_weight",
            woreda_data=indep_subset,
            woreda_x_col="indep_coverage",
            woreda_region_col="region",
            woreda_year_col="year",
            region_col=region_fe_col,
            dataset_col=dataset_col,
            woreda_weight_col="denominator_population",
        )
    else:
        # Standard beta regression for zonal/woreda level or when ecological is disabled
        result = stats_utils.regression(
            group,
            x_col="dhis2_coverage",
            y_col="survey_coverage",
            weight_col="combined_weight",
            method="beta",
            region_col=region_fe_col,
            dataset_col=dataset_col,
        )

    r2_weighted[dhis2_ind] = result["r2"]
    regression_results_cache[dhis2_ind] = result

    # --- Metrics ---
    full_res = result.get("results")
    p_val, z_score = np.nan, np.nan
    if full_res is not None:
        targets = ["x1", "dhis2_coverage"]
        # Try getting from exog_names (most reliable for arrays)
        names = getattr(full_res.model, "exog_names", [])
        idx = next((i for i, n in enumerate(names) if n in targets), -1)

        assert idx != -1, f"Could not find {targets} in {names}"
        p_val = np.array(full_res.pvalues)[idx]
        z_score = np.array(full_res.tvalues)[idx]

    ps_r2_imp = np.nan
    r2_full = np.nan
    r2_red = np.nan

    if full_res:
        assert len(group) > 0, "Group is empty, cannot compute reduced model R2."
        dp_red = [pd.Series(1.0, index=group.index, name="const")]
        if dataset_col:
            dp_red.append(
                pd.get_dummies(
                    group[dataset_col], prefix="d", drop_first=True, dtype=float
                )
            )
        if region_fe_col:
            dp_red.append(
                pd.get_dummies(
                    group[region_fe_col], prefix="r", drop_first=True, dtype=float
                )
            )
        X_red = pd.concat(dp_red, axis=1)

        y_red = group["survey_coverage"]
        w_red = group["combined_weight"]
        r2_full = result["r2"]
        r2_red = 0.0

        if "beta" in result["method"]:
            mr = stats_utils.WeightedBetaModel(
                y_red.clip(0.01, 0.99), X_red, weights=w_red
            )
            rr = mr.fit(disp=0)

            if result["method"] == "ecological_beta":
                # Ecological uses SSE-based R2; match it for reduced model
                sst = ((y_red - y_red.mean()) ** 2).sum()
                sse = ((y_red - rr.fittedvalues) ** 2).sum()
                r2_red = 1 - sse / sst
            else:
                r2_red = rr.prsquared
        else:
            mr = sm.WLS(y_red, X_red, weights=w_red)
            rr = mr.fit()
            r2_red = rr.rsquared
        ps_r2_imp = r2_full - r2_red

    dc = result.get("dataset_coefs", {})

    be = result[
        "b"
    ]  # + (np.mean(list(dc.values())) if dc else 0) # We thought averaging fixed effects would be better, but DHS gives lower distances overall

    def pd_func(x):
        v = result["m"] * x + be
        return expit(v) if "beta" in result["method"] else v

    dist_id, _ = quad(lambda x: (pd_func(x) - x) ** 2, 0, 1)

    # Calculate Quality Score
    quality = np.nan
    if not np.isnan(ps_r2_imp) and not np.isnan(dist_id):
        j_min = 0.000833709
        j_max = 7 / 12

        # Term 1: Partial Pseudo R2
        term1 = (r2_full - r2_red) / (1 - r2_red) if (1 - r2_red) > 1e-9 else 0
        term1 = max(0, term1)

        # Term 2: Normalized Distance
        term2 = 1 - (dist_id - j_min) / (j_max - j_min)
        term2 = max(0, min(1, term2))

        quality = 0.5 * term1 + 0.5 * term2

    fit_metrics.append(
        {
            "indicator": dhis2_ind,
            "p_value": p_val,
            "z_score": z_score,
            "ps_r2_imp": ps_r2_imp,
            "dist_id": dist_id,
            "quality": quality,
        }
    )

    if (
        COMPUTE_COOKS_DISTANCE
        and dhis2_ind == "MAT_Skilled Birth Attendance"
        and ADMIN_LEVEL == "regional"
    ):
        from scipy.special import expit

        print(f"Computing Cook's Distance for {dhis2_ind}...")

        def predict(r, res):
            val = r["dhis2_coverage"] * res["m"] + res["b"]
            if dataset_col and "dataset_coefs" in res:
                val += res["dataset_coefs"].get(r[dataset_col], 0)
            if region_fe_col and "region_coefs" in res:
                val += res["region_coefs"].get(r[region_fe_col], 0)
            return expit(val) if "beta" in res["method"] else val

        yhat = group.apply(lambda r: predict(r, result), axis=1)
        k = 2 + (len(result.get("dataset_coefs", {})) - 1 if dataset_col else 0)
        mse = ((group["survey_coverage"] - yhat) ** 2).sum() / (len(group) - k)

        cooks = []
        for idx in group.index:
            sub = group.drop(idx)
            if use_eco:
                loo = stats_utils.regression_ecological(
                    sub,
                    "dhis2_coverage",
                    "survey_coverage",
                    "combined_weight",
                    indep_subset,
                    "indep_coverage",
                    "region",
                    "year",
                    region_col=region_fe_col,
                    dataset_col=dataset_col,
                    woreda_weight_col="denominator_population",
                    print_summary=False,
                )
            else:
                loo = stats_utils.regression(
                    sub,
                    "dhis2_coverage",
                    "survey_coverage",
                    "combined_weight",
                    method="beta",
                    region_col=region_fe_col,
                    dataset_col=dataset_col,
                    print_summary=False,
                )
            yhat_loo = group.apply(lambda r: predict(r, loo), axis=1)
            cooks.append(((yhat - yhat_loo) ** 2).sum() / (k * mse))

        cooks_df = pd.DataFrame(
            {
                "region": group["region"],
                "source": group["source"],
                "year": group["year"],
                "cooks": cooks,
            }
        ).sort_values("cooks", ascending=False)
        print(cooks_df)
        print(f"Mean Cook's distance: {cooks_df['cooks'].mean()}")

    # Save parameters only if this indicator exists in DHS map
    if dhis2_ind in dhs_mapped_indicators:
        dhs_keys = [k for k, v in DHS_INDICATOR_MAP.items() if v == dhis2_ind]
        for dhs_key in dhs_keys:
            params = {
                "r2": result["r2"],
                "slope": result["m"],
                "intercept": result["b"],
                "region_coefs": result.get("region_coefs", {}),
                "dataset_coefs": result.get("dataset_coefs", {}),
                "dhis2_indicator": dhis2_ind,
                "method": result["method"],
                "admin_level": ADMIN_LEVEL,
                "admin_level_indep": ADMIN_LEVEL_INDEP,
                "ecological": use_eco,
            }

            regression_params[dhs_key] = params

params_stem = "dhs_dhis2_calibration_params"
if ADMIN_LEVEL_INDEP == "zonal" and ADMIN_LEVEL == "regional":
    params_stem += "_zonal_indep"
params_path = Path(f"estimates/{params_stem}.json")
with open(params_path, "w") as f:
    json.dump(regression_params, f, indent=2)
print(
    f"Saved calibration parameters for {len(regression_params)} indicators to {params_path}"
)
print(f"  Admin level: {ADMIN_LEVEL}")

# Sort
merged = merged.sort_values(["dhis2_indicator"] + UNIT_COLS + ["year"])

# Save
out_path = Path("estimates/dhs_pma_dhis2_coverage_comparison.csv")
merged.to_csv(out_path, index=False)

print(f"Wrote {len(merged):,} rows to {out_path}")
print(
    f"{merged['dhis2_indicator'].nunique()} indicators, {merged[UNIT_LABEL].nunique()} {UNIT_LABEL}s"
)
print(f"Mean abs difference: {merged['abs_difference'].mean():.3f}")
print(f"Correlation: {merged['survey_coverage'].corr(merged['dhis2_coverage']):.3f}")
print("\nWeighted R² (by DHIS2 indicator, weighted by combined variance):")
for indicator in sorted(r2_weighted.keys()):
    print(f"  {indicator}: {r2_weighted[indicator]:.3f}")

fig_dir = Path("figures")
fig_dir.mkdir(exist_ok=True)

marker_map = {"DHS": "o", "PMA": "x", "Metaanalysis": "^"}

for dhis2_ind in sorted(merged["dhis2_indicator"].unique()):
    subset = merged[merged["dhis2_indicator"] == dhis2_ind]

    r2_val = r2_weighted.get(dhis2_ind, float("nan"))
    surveys = subset["source"].unique()
    title = f"{dhis2_ind}\nSources: {', '.join(surveys)} | Weighted R² = {r2_val:.3f} | Level: {ADMIN_LEVEL}"

    safe_name = (
        dhis2_ind.replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("<", "lt")
    )
    out_file = fig_dir / f"survey_comparison_{safe_name}.png"

    y_label_text = "Survey Coverage (DHS/PMA)"

    reg_res = regression_results_cache.get(dhis2_ind)
    reg_line_params = None
    if reg_res:
        reg_line_params = {
            "slope": reg_res["m"],
            "intercept": reg_res["b"],
            "method": reg_res["method"],
        }

    stats_utils.plot_scatter_comparison(
        data=subset,
        x_col="dhis2_coverage",
        y_col="survey_coverage",
        label_col=UNIT_LABEL,
        year_col="year",
        title=title,
        x_label=f"DHIS2: {dhis2_ind}",
        y_label=y_label_text,
        output_path=out_file,
        weight_col="combined_weight",
        size_scale_factor=250,
        marker_col="source",
        marker_map=marker_map,
        regression_line_params=reg_line_params,
        color_col="region",  # Always color by region, even for zonal/woreda level
    )

print(f"Saved {merged['dhis2_indicator'].nunique()} scatterplots to {fig_dir}/")

print(
    f"\n{'Indicator':<40} | {'p-val':<8} | {'z-score':<8} | {'R2 Imp':<8} | {'Dist Id':<8} | {'Quality':<8}"
)
print("-" * 96)
for m in sorted(fit_metrics, key=lambda x: x["quality"], reverse=True):
    print(
        f"{m['indicator'][:40]:<40} | {(f'{m['p_value']:.2e}' if pd.notna(m['p_value']) else 'nan'):<8} | {(f'{m['z_score']:.2f}' if pd.notna(m['z_score']) else 'nan'):<8} | {(f'{m['ps_r2_imp']:.3f}' if pd.notna(m['ps_r2_imp']) else 'nan'):<8} | {m['dist_id']:.4f}   | {(f'{m['quality']:.3f}' if pd.notna(m['quality']) else 'nan'):<8}"
    )

# Calculate rank correlation
df_metrics = pd.DataFrame(fit_metrics)
if (
    not df_metrics.empty
    and "z_score" in df_metrics.columns
    and "quality" in df_metrics.columns
):
    corr_zq = df_metrics["z_score"].corr(df_metrics["quality"], method="spearman")
    print(f"\nSpearman Rank Correlation (z-score vs Quality): {corr_zq:.3f}")

    # Correlations with Negative Control status
    # Identify negative controls by name
    df_metrics["is_neg"] = (
        df_metrics["indicator"].str.contains("Negative Control", case=False).astype(int)
    )

    # Only compute if we have variation (both negative controls and non-negative controls)
    if df_metrics["is_neg"].nunique() > 1:
        corr_q_neg = df_metrics["quality"].corr(df_metrics["is_neg"], method="spearman")
        corr_z_neg = df_metrics["z_score"].corr(df_metrics["is_neg"], method="spearman")

        print(
            f"Spearman Rank Correlation (Quality vs Is_Negative_Control): {corr_q_neg:.3f}"
        )
        print(
            f"Spearman Rank Correlation (z-score vs Is_Negative_Control): {corr_z_neg:.3f}"
        )
