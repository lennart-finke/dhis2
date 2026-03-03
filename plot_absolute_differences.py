import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.path import Path as MplPath

# output path
OUTPUT_DIR = Path("figures/residuals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGION_NAME_MAP = {
    "Benishangul Gumuz": "Benishangul Gumz",
    "Benishangul-Gumuz": "Benishangul Gumz",
    "Oromiya": "Oromia",
    "SNNPR": "SNNP",
    "Southern Nations, Nationalities, and Peoples": "SNNP",
}

# Regions that were formerly part of SNNP — must match compute_calibrated_coverage.py
SNNP_SUBREGIONS = [
    "Sidama",
    "South West Ethiopia",
    "South Ethiopia",
    "Central Ethiopia",
    "SNNP",
]

LEVELS = [
    {
        "name": "regional",
        "calib_file": Path("estimates/regional_calibrated_coverage_geojson.csv"),
        "dhs_file": Path("estimates/dhs_regional_estimates.csv"),
        "pma_file": Path("estimates/pma_regional_estimates.csv"),
        "boundary_path": Path(
            "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin1.geojson"
        ),
        "id_col": "adm1_name",
        "data_col": "region",
        "join_cols": ["region", "year", "indicator"],
        "aggregate_snnp": True,
    },
    {
        "name": "zonal",
        "calib_file": Path("estimates/zonal_calibrated_coverage_geojson.csv"),
        "dhs_file": Path("estimates/dhs_zonal_estimates.csv"),
        "pma_file": Path("estimates/pma_zonal_estimates.csv"),
        "boundary_path": Path(
            "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin2.geojson"
        ),
        "id_col": "adm2_name",
        "data_col": "zone",
        "join_cols": ["region", "zone", "year", "indicator"],
        "aggregate_snnp": False,
    },
    {
        "name": "woreda",
        "calib_file": Path("estimates/woreda_calibrated_coverage_geojson.csv"),
        "dhs_file": Path("estimates/dhs_woreda_estimates.csv"),
        "pma_file": Path("estimates/pma_woreda_estimates.csv"),
        "boundary_path": Path(
            "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin3.geojson"
        ),
        "id_col": "adm3_name",
        "data_col": "woreda",
        "join_cols": ["region", "zone", "woreda", "year", "indicator"],
        "aggregate_snnp": False,
    },
]


def normalize_regions(df, region_col="region"):
    """Normalize region names to match GeoJSON."""
    if df.empty or region_col not in df.columns:
        return df
    df[region_col] = df[region_col].replace(REGION_NAME_MAP)
    return df


def aggregate_snnp_calib(df):
    """Aggregate SNNP sub-regions for calibrated data."""
    if df.empty:
        return df

    # Map sub-regions to SNNP
    df.loc[df["region"].isin(SNNP_SUBREGIONS), "region"] = "SNNP"

    group_cols = ["region", "year", "indicator"]

    if "dhis2_value" not in df.columns or "estimated_denominator" not in df.columns:
        return df

    df["calibrated_numerator"] = df["calibrated_coverage"] * df["estimated_denominator"]

    agg_df = (
        df.groupby(group_cols)
        .agg(
            {
                "dhis2_value": "sum",
                "estimated_denominator": "sum",
                "calibrated_numerator": "sum",
                "dhis2_indicator": "first",
            }
        )
        .reset_index()
    )

    agg_df["dhis2_coverage"] = agg_df["dhis2_value"] / agg_df["estimated_denominator"]
    agg_df["calibrated_coverage"] = agg_df["calibrated_numerator"] / agg_df["estimated_denominator"]

    return agg_df


def aggregate_snnp_survey(df):
    """Aggregate SNNP sub-regions for survey data."""
    if df.empty:
        return df

    df.loc[df["region"].isin(SNNP_SUBREGIONS), "region"] = "SNNP"

    df = df.dropna(subset=["coverage", "denominator"])
    df["numerator"] = df["coverage"] * df["denominator"]

    group_cols = ["region", "year", "indicator", "source"]

    agg_df = (
        df.groupby(group_cols)
        .agg({"numerator": "sum", "denominator": "sum"})
        .reset_index()
    )

    agg_df["coverage"] = agg_df["numerator"] / agg_df["denominator"]

    return agg_df


def calculate_bin_variance(p, n):
    p = np.clip(p, 0, 1)
    return p * (1 - p) / n


def get_polygon_patches(geometry):
    patches = []
    if geometry.geom_type == "Polygon":
        polygons = [geometry]
    elif geometry.geom_type == "MultiPolygon":
        polygons = geometry.geoms
    else:
        return []

    for polygon in polygons:
        vertices = []
        codes = []
        for ring in [polygon.exterior] + list(polygon.interiors):
            coords = list(ring.coords)
            if not coords:
                continue
            vertices.append(coords[0])
            codes.append(MplPath.MOVETO)
            for pt in coords[1:-1]:
                vertices.append(pt)
                codes.append(MplPath.LINETO)
            vertices.append(coords[0])
            codes.append(MplPath.CLOSEPOLY)

        if vertices:
            path = MplPath(vertices, codes)
            patches.append(PathPatch(path))

    return patches


def plot_map(
    data_df, boundary_path, value_col, id_col, data_col, output_path, level_name
):
    """
    Plot a choropleth map of the values.
    """
    # Load boundaries
    gdf = gpd.read_file(boundary_path)

    data_map = dict(zip(data_df[data_col], data_df[value_col]))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    patches = []
    values = []
    unmatched_patches = []

    for idx, row in gdf.iterrows():
        geo_id = row[id_col]
        poly_patches = get_polygon_patches(row["geometry"])

        # Special handling for SNNP subregions only if we aggregated them (Regional level)
        lookup_key = geo_id
        if level_name == "regional" and geo_id in SNNP_SUBREGIONS:
            lookup_key = "SNNP"

        if lookup_key in data_map:
            val = data_map[lookup_key]
            if pd.notna(val):
                patches.extend(poly_patches)
                values.extend([val] * len(poly_patches))
            else:
                unmatched_patches.extend(poly_patches)
        else:
            unmatched_patches.extend(poly_patches)

    if patches:
        collection = PatchCollection(
            patches, cmap="Reds", edgecolor="black", linewidth=0.1
        )
        collection.set_array(np.array(values))
        ax.add_collection(collection)
        cbar = plt.colorbar(collection, ax=ax, shrink=0.6)
        cbar.set_label("Mean Standardized Residual")
        collection.set_clim(0, 20)

    if unmatched_patches:
        u_collection = PatchCollection(
            unmatched_patches, facecolor="lightgray", edgecolor="black", linewidth=0.1
        )
        ax.add_collection(u_collection)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved map to {output_path}")


def process_level(config):
    level_name = config["name"]
    print(f"\n--- Processing {level_name.upper()} level ---")

    assert config["calib_file"].exists()
    assert config["dhs_file"].exists()
    assert config["pma_file"].exists()

    calib_df = pd.read_csv(config["calib_file"])
    dhs_df = pd.read_csv(config["dhs_file"])
    pma_df = pd.read_csv(config["pma_file"])

    dhs_df["source"] = "DHS"
    pma_df["source"] = "PMA"

    # Normalize regions
    calib_df = normalize_regions(calib_df)
    dhs_df = normalize_regions(dhs_df)
    pma_df = normalize_regions(pma_df)

    # Map PMA indicators to standard names
    PMA_NAME_MAP = {
        "Mother dewormed during pregnancy": "Drugs for Intestinal Parasites",
        "0-12 months Pentavalent3": "DPT3/Penta3",
        "0-12 months Measles1": "Measles 1st Dose",
        "SBA": "Skilled Birth Attendance",
        "Post-partum had ANC": "ANC First Visit",
    }
    pma_df["indicator"] = pma_df["indicator"].replace(PMA_NAME_MAP)

    # Aggregate SNNP if required
    if config["aggregate_snnp"]:
        print("Aggregating SNNP sub-regions...")
        calib_df = aggregate_snnp_calib(calib_df)
        dhs_df = aggregate_snnp_survey(dhs_df)
        pma_df = aggregate_snnp_survey(pma_df)

    survey_df = pd.concat([dhs_df, pma_df], ignore_index=True)

    # Merge calibrated estimates with survey data
    # The calibrated_coverage column is already computed by compute_calibrated_coverage.py
    merged = pd.merge(
        calib_df, survey_df, on=config["join_cols"], suffixes=("_model", "_survey")
    )

    if merged.empty:
        print("No matching records found.")
        return

    print(f"Comparing {len(merged)} matching records.")

    # Calc Stats — use pre-computed calibrated_coverage directly
    merged["var_model"] = calculate_bin_variance(
        merged["calibrated_coverage"], merged["estimated_denominator"]
    )
    merged["var_survey"] = calculate_bin_variance(
        merged["coverage"], merged["denominator"]
    )

    merged["abs_diff"] = (merged["calibrated_coverage"] - merged["coverage"]).abs()
    merged["combined_se"] = np.sqrt(merged["var_model"] + merged["var_survey"])

    safe_mask = merged["combined_se"] > 0
    merged.loc[safe_mask, "z_score"] = (
        merged.loc[safe_mask, "abs_diff"] / merged.loc[safe_mask, "combined_se"]
    )
    merged.loc[(~safe_mask) & (merged["abs_diff"] == 0), "z_score"] = 0
    merged.loc[(~safe_mask) & (merged["abs_diff"] > 0), "z_score"] = np.nan

    valid_z = merged.dropna(subset=["z_score"])
    print(f"Calculated Z-scores for {len(valid_z)} records.")

    # Plot per indicator
    unique_indicators = valid_z["indicator"].unique()
    level_out_dir = OUTPUT_DIR / level_name
    level_out_dir.mkdir(exist_ok=True)

    regional_summary = []

    for indicator in unique_indicators:
        subset = valid_z[valid_z["indicator"] == indicator]

        # Aggregate by spatial unit (data_col) - average z-score over years
        stats = subset.groupby(config["data_col"])["z_score"].mean().reset_index()

        if level_name == "regional":
            regional_summary.append(
                stats.set_index(config["data_col"])["z_score"].rename(indicator)
            )

        safe_ind = (
            indicator.replace("/", "_")
            .replace(" ", "_")
            .replace("<", "lt")
            .replace(">", "gt")
            .lower()
        )
        output_file = level_out_dir / f"residuals_{safe_ind}.png"

        print(f"Plotting {indicator}...")

        plot_map(
            data_df=stats,
            boundary_path=config["boundary_path"],
            value_col="z_score",
            id_col=config["id_col"],
            data_col=config["data_col"],
            output_path=output_file,
            level_name=level_name,
        )

    if level_name == "regional" and regional_summary:
        print("\n--- Regional Residuals Table (Mean Absolute Z-Score) ---")
        full_table = pd.concat(regional_summary, axis=1)

        col_map = {
            "C-section": "C-Sections",
            "Skilled Birth Attendance": "SBA",
            "Measles 1st Dose": "Measles1",
            "MAM": "MAM",
            "DPT3/Penta3": "DPT3",
            "Drugs for Intestinal Parasites": "Deworming",
            "Low Birth Weight": "LBW",
            "ANC First Visit": "ANC"
        }
        full_table = full_table.rename(columns=col_map)
        
        # Ensure all expected columns are present
        expected_cols = ["ANC", "C-Sections", "SBA", "Measles1", "MAM", "DPT3", "LBW", "Deworming"]
        for col in expected_cols:
            if col not in full_table.columns:
                full_table[col] = np.nan
                
        # Calculate mean before dropping/reordering, or after? Original table probably took mean over the ones present
        full_table["Mean"] = full_table.mean(axis=1)
        full_table.loc["Mean"] = full_table.mean(axis=0)

        # Reorder
        ordered_cols = expected_cols + ["Mean"]
        full_table = full_table[ordered_cols]

        print("```latex")
        print("\\begin{landscape}")
        print("    \\begin{table}[]")
        print("    \\centering")
        columns = full_table.columns.tolist()
        columns_str = " & ".join(columns[:-1]) + " & \\bf Mean"
        print(f"    \\begin{{tabular}}{{rl|l{'r'*(len(columns)-1)}|r}}")
        print("         & & \\bf Indicator \\\\")
        print(f"         & & {columns_str} \\\\")
        print("         \\hline")
        print("        \\bf Region")
        for idx, row in full_table.iterrows():
            if idx == "Mean":
                continue
            row_str = " & ".join([f"{v:.2f}" if pd.notna(v) else "-" for v in row])
            print(f"         & {idx:<18} & {row_str} \\\\")
        
        # Then the mean row
        mean_row_str = " & ".join([f"{v:.2f}" if pd.notna(v) else "-" for v in full_table.loc["Mean"]])
        print(f"         \\bf Mean              &      & {mean_row_str} \\\\")
        print("    \\end{tabular}")
        print("    \\caption{Standardized residuals for the ecological beta regression model predicting survey SBA coverage from DHIS2 indicators, by region. All former SNNP regions were aggregated and treated as one for the purpose of this calculation. Compare also \\Cref{tab:cooks}.}")
        print("    \\label{tab:residuals}")
        print("\\end{table}")
        print("\\end{landscape}")
        print("```")


def main():
    for level_config in LEVELS:
        process_level(level_config)


if __name__ == "__main__":
    main()
