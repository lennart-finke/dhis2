"""
Compute calibrated coverage estimates from DHIS2 data at all administrative levels.
Applies DHS-DHIS2 calibration parameters to get estimated "true" coverage.
Processes regional, zonal, and woreda levels in a single run.
Saves to level-specific output files.
"""

import pandas as pd
import json
from pathlib import Path
import stats_utils
import argparse

# Load calibration parameters
params_path = Path("estimates/dhs_dhis2_calibration_params.json")
with open(params_path) as f:
    calibration_params = json.load(f)

# Load zonal independent parameters if they exist
params_path_zonal = Path("estimates/dhs_dhis2_calibration_params_zonal_indep.json")
if params_path_zonal.exists():
    with open(params_path_zonal) as f:
        calibration_params_zonal = json.load(f)
else:
    calibration_params_zonal = calibration_params

# Parse args
parser = argparse.ArgumentParser(
    description="Compute calibrated coverage at all admin levels"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="PMA",
    help="Dataset to emulate for predictions (e.g., DHS, PMA)",
)
args = parser.parse_args()

# Indicators to calibrate
dhis2_to_dhs = {}
for dhs_ind, params in calibration_params.items():
    dhis2_to_dhs[params["dhis2_indicator"]] = dhs_ind

# Regions that were formerly part of SNNP and should share its fixed effect
FORMER_SNNP_REGIONS = {
    "Sidama",
    "South West Ethiopia",
    "South Ethiopia",
    "Central Ethiopia",
}

# Define administrative levels to process
admin_levels = [
    {
        "name": "regional",
        "input_path": Path("estimates/dhis2_coverage_estimates_gps.csv"),
        "pivot_cols": ["region", "year"],
        "output_suffix": "regional",
        "has_zone": False,
        "has_woreda": False,
        "has_kebele": False,
    },
    {
        "name": "zonal",
        "input_path": Path("estimates/dhis2_coverage_estimates_zonal_gps.csv"),
        "pivot_cols": ["region", "zone", "year"],
        "output_suffix": "zonal",
        "has_zone": True,
        "has_woreda": False,
        "has_kebele": False,
    },
    {
        "name": "woreda",
        "input_path": Path("estimates/dhis2_coverage_estimates_woreda_gps.csv"),
        "pivot_cols": ["region", "zone", "woreda", "year"],
        "output_suffix": "woreda",
        "has_zone": True,
        "has_woreda": True,
        "has_kebele": False,
    },
    {
        "name": "kebele",
        "input_path": Path("estimates/dhis2_coverage_estimates_kebele_gps.csv"),
        "pivot_cols": ["region", "zone", "woreda", "kebele", "year"],
        "output_suffix": "kebele",
        "has_zone": True,
        "has_woreda": True,
        "has_kebele": True,
    },
]

# Process each administrative level
for level_config in admin_levels:
    print(f"Processing {level_config['name'].upper()} level")

    # Load data
    assert level_config["input_path"].exists()

    df = pd.read_csv(level_config["input_path"])

    # Filter for years of interest
    df = df[df["year"].between(2016, 2025)]

    # Handle zone column naming variations
    if level_config["has_zone"]:
        if "zone" not in df.columns and "zone_geojson" in df.columns:
            df.rename(columns={"zone_geojson": "zone"}, inplace=True)

    # Build indexed dataframe
    pivot_cols = level_config["pivot_cols"]
    df_indexed = df.set_index(pivot_cols + ["indicator"]).sort_index()

    # Prepare results
    calibrated_records = []

    # Iterate through the dataframe rows
    # Each row represents one (region, [zone], [woreda], year, indicator)
    for idx, row in df.iterrows():
        dhis2_ind = row["indicator"]

        # Check if this is a primary indicator for calibration
        if dhis2_ind not in dhis2_to_dhs:
            continue

        dhs_ind = dhis2_to_dhs[dhis2_ind]
        if level_config["name"] == "zonal" and dhs_ind in calibration_params_zonal:
            params = calibration_params_zonal[dhs_ind]
        else:
            params = calibration_params[dhs_ind]

        # Extract values
        raw_cov = row["coverage"]
        numerator = row["value"]
        denominator = row["denominator_population"]
        region = row["region"]

        # Check for NaN
        if pd.isna(raw_cov):
            continue

        # Apply calibration
        region_coefs = params.get("region_coefs", {})
        dataset_coefs = params.get("dataset_coefs", {})

        # Use SNNP fixed effect for former SNNP regions
        calc_region = region
        if region in FORMER_SNNP_REGIONS:
            calc_region = "SNNP"

        calibrated_cov = stats_utils.predict_coverage(
            raw_cov,
            params["slope"],
            params["intercept"],
            params["method"],
            region=calc_region,
            region_coefs=region_coefs,
            dataset=args.dataset,
            dataset_coefs=dataset_coefs,
        )

        # Build record with appropriate administrative columns
        record = {
            "region": region,
            "year": row["year"],
            "indicator": dhs_ind,
            "dhis2_indicator": dhis2_ind,
            "dhis2_value": numerator,
            "estimated_denominator": denominator,
            "dhis2_coverage": raw_cov,
            "calibrated_coverage": calibrated_cov,
            "calibration_slope": params["slope"],
            "calibration_intercept": params["intercept"],
            "calibration_r2": params["r2"],
        }

        # Add zone and woreda columns if applicable
        if level_config["has_zone"]:
            record["zone"] = row["zone"]
        if level_config["has_woreda"]:
            record["woreda"] = row["woreda"]
        if level_config["has_kebele"]:
            record["kebele"] = row["kebele"]

        calibrated_records.append(record)

    calibrated_df = pd.DataFrame(calibrated_records)

    # Save with level-specific filenames
    output_suffix = level_config["output_suffix"]
    output_file = f"estimates/{output_suffix}_calibrated_coverage_geojson.csv"

    calibrated_df.to_csv(output_file, index=False)
    print(f"Wrote {len(calibrated_df):,} rows to {output_file}")
    
    # Save latest year (2024) for validation scripts
    latest_df = calibrated_df[calibrated_df["year"] == 2024]
    latest_file = f"estimates/{output_suffix}_calibrated_coverage_geojson_latest.csv"
    latest_df.to_csv(latest_file, index=False)
    print(f"Wrote {len(latest_df):,} rows to {latest_file}")
    
    print(f"Indicators processed: {sorted(calibrated_df['indicator'].unique())}")

print("All administrative levels processed successfully!")
