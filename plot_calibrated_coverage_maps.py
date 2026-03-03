"""
Plot calibrated coverage estimates on maps of Ethiopia at all administrative levels.
Uses GeoJSON admin boundaries and creates choropleth maps for regional, zonal, and woreda levels.
Saves figures to figures/maps/
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.collections import PatchCollection
import argparse

DEBUG = False
PLOT_YEAR = 2024

# Parse arguments
parser = argparse.ArgumentParser(description="Plot coverage maps at all admin levels")
args = parser.parse_args()

# Define administrative levels to process
admin_levels = [
    {
        "name": "regional",
        "coverage_file": "estimates/regional_calibrated_coverage_geojson.csv",
        "geojson_path": "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin1.geojson",
        "shapefile_path": None,  # Add if available
        "id_col_geojson": "adm1_name",
        "id_col_shapefile": None,
        "data_col": "region",
    },
    {
        "name": "zonal",
        "coverage_file": "estimates/zonal_calibrated_coverage_geojson.csv",
        "geojson_path": "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin2.geojson",
        "shapefile_path": None,
        "id_col_geojson": "adm2_name",
        "id_col_shapefile": None,
        "data_col": "zone",
    },
    {
        "name": "woreda",
        "coverage_file": "estimates/woreda_calibrated_coverage_geojson.csv",
        "geojson_path": "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin3.geojson",
        "shapefile_path": "data/geography/Ethiopia_adm3_uscb_2016.shp",
        "id_col_geojson": "adm3_name",
        "id_col_shapefile": "GEO_MATCH",
        "data_col": "woreda",
    },
    {
        "name": "kebele",
        "coverage_file": "estimates/kebele_calibrated_coverage_geojson.csv",
        "geojson_path": "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin4.geojson",
        "shapefile_path": None,
        "id_col_geojson": "RK_NAME",
        "id_col_shapefile": None,
        "data_col": "kebele",
    },
]

# Create output directory
fig_dir = Path("figures/maps")
fig_dir.mkdir(parents=True, exist_ok=True)


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

        # Iterate over exterior and all interiors (holes)
        for ring in [polygon.exterior] + list(polygon.interiors):
            coords = list(ring.coords)
            if not coords:
                continue

            # Start of ring
            vertices.append(coords[0])
            codes.append(MplPath.MOVETO)

            # Segments
            for pt in coords[1:-1]:
                vertices.append(pt)
                codes.append(MplPath.LINETO)

            # Close ring
            vertices.append(coords[0])  # Last point for closure
            codes.append(MplPath.CLOSEPOLY)

        if vertices:
            path = MplPath(vertices, codes)
            # Create PathPatch. Note: facecolor/edgecolor will be set by PatchCollection
            patches.append(PathPatch(path))

    return patches


def plot_coverage_map(
    coverage_data,
    boundary_features,
    indicator,
    output_path,
    data_col,
    shapefile_id_col,
    title=None,
    background_boundary_features=None,
):
    # Directly read in the admin unit's coverage
    admin_coverage = dict(
        zip(coverage_data[data_col], coverage_data["calibrated_coverage"])
    )

    matched = 0
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    all_patches, all_values = [], []
    unmatched_patches = []
    out_of_range_patches = []

    if background_boundary_features:
        bg_patches = []
        for bg_id, row in background_boundary_features.items():
            bg_patches.extend(get_polygon_patches(row["geometry"]))
        if bg_patches:
            bg_collection = PatchCollection(
                bg_patches, facecolor="lightgray", edgecolor="black", linewidth=0.1
            )
            ax.add_collection(bg_collection)

    for shapefile_id, row in boundary_features.items():
        patches = get_polygon_patches(row["geometry"])

        if DEBUG:
            # Plot admin unit name at centroid
            centroid = row["geometry"].centroid
            ax.text(
                centroid.x,
                centroid.y,
                str(shapefile_id),
                fontsize=4,
                ha="center",
                va="center",
                color="black",
                clip_on=True,
            )

        if shapefile_id in admin_coverage:
            matched += 1
            value = admin_coverage[shapefile_id]
            if pd.notna(value):
                for patch in patches:
                    if 0 <= value <= 1:
                        all_patches.append(patch)
                        all_values.append(value)
                    else:
                        out_of_range_patches.append(patch)
            else:
                for patch in patches:
                    unmatched_patches.append(patch)
        else:
            for patch in patches:
                unmatched_patches.append(patch)

    print(f"  Matched {matched} administrative units for {indicator}")

    # Determine colormap based on indicator type
    bad_keywords = [
        "wasting",
        "malnutrition",
        "mam",
        "sam",
        "low birth weight",
        "mortality",
        "stunting",
        "underweight",
    ]
    rare_keywords = [
        "c-section",
        "c section",
        "intestinal parasites",
        "low birth weight",
        "wasting",
        "mam",
    ]
    is_bad = any(k in indicator.lower() for k in bad_keywords)
    is_rare = any(k in indicator.lower() for k in rare_keywords)

    cmap_name = "RdYlGn_r" if is_bad else "RdYlGn"
    vmax = 0.5 if is_rare else 1.0

    if all_patches:
        collection = PatchCollection(
            all_patches, cmap=cmap_name, edgecolor="black", linewidth=0.1
        )
        collection.set_array(np.array(all_values))
        collection.set_clim(0, vmax)
        ax.add_collection(collection)
        cbar = plt.colorbar(collection, ax=ax, shrink=0.6, label="Estimated Coverage")
        cbar.ax.tick_params(labelsize=8)

    if out_of_range_patches:
        oor_collection = PatchCollection(
            out_of_range_patches, facecolor="black", edgecolor="black", linewidth=0.1
        )
        ax.add_collection(oor_collection)

    if unmatched_patches:
        unmatched_collection = PatchCollection(
            unmatched_patches, facecolor="lightgray", edgecolor="black", linewidth=0.1
        )
        ax.add_collection(unmatched_collection)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.set_title(
        title or f"Estimated Coverage: {indicator}", fontsize=11, fontweight="bold"
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    if data_col != "kebele":
        # Also save as pdf in pdf subfolder
        pdf_path = output_path.parent / "pdf" / output_path.name.replace(".png", ".pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {output_path}")


woreda_boundary_features = None

# Process each administrative level
for level_config in admin_levels:
    print(f"Processing {level_config['name'].upper()} level")

    # Determine which boundary file to use
    boundary_path = Path(level_config["geojson_path"])
    shapefile_id_col = level_config["id_col_geojson"]

    # Load boundaries
    shp_gdf = gpd.read_file(boundary_path)

    # Load coverage data
    coverage_file = Path(level_config["coverage_file"])
    if not coverage_file.exists():
        print(f"  Coverage file not found: {coverage_file}")
        continue

    coverage_df_full = pd.read_csv(coverage_file)
    coverage_df = coverage_df_full[coverage_df_full["year"] == PLOT_YEAR]

    if coverage_df.empty:
        print(f"  No data found for year {PLOT_YEAR} in {coverage_file}")
        continue

    # Build boundary lookup
    boundary_features = {}
    for idx, row in shp_gdf.iterrows():
        geouid = row[shapefile_id_col]
        if geouid:
            boundary_features[geouid] = row

    if level_config["data_col"] == "woreda":
        woreda_boundary_features = boundary_features

    print(f"  Loaded {len(boundary_features)} boundaries from {boundary_path.name}")
    print(
        f"  Coverage data has {coverage_df[level_config['data_col']].nunique()} unique units"
    )

    bg_features = None
    if level_config["data_col"] == "kebele":
        bg_features = woreda_boundary_features

    # ANC data is not available after 2022; use 2021 values when plotting later years.
    ANC_MAX_YEAR = 2021

    # Build the indicator set: union of PLOT_YEAR indicators + any ANC indicators from
    # ANC_MAX_YEAR. This is necessary because when PLOT_YEAR > ANC_MAX_YEAR, ANC rows
    # are absent from coverage_df so the ANC indicator would never enter the loop.
    plot_indicators = set(coverage_df["indicator"].unique())
    if PLOT_YEAR > ANC_MAX_YEAR:
        anc_fallback_indicators = {
            ind
            for ind in coverage_df_full[coverage_df_full["year"] == ANC_MAX_YEAR][
                "indicator"
            ].unique()
            if "anc" in ind.lower()
        }
        plot_indicators |= anc_fallback_indicators

    # Create maps for each indicator
    for indicator in sorted(plot_indicators):
        is_anc = "anc" in indicator.lower()
        if is_anc and PLOT_YEAR > ANC_MAX_YEAR:
            subset = coverage_df_full[
                (coverage_df_full["indicator"] == indicator)
                & (coverage_df_full["year"] == ANC_MAX_YEAR)
            ]
            if subset.empty:
                print(
                    f"  No {ANC_MAX_YEAR} data found for ANC indicator '{indicator}', skipping."
                )
                continue
        else:
            subset = coverage_df[coverage_df["indicator"] == indicator]
        safe_name = (
            indicator.replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )
        output_path = fig_dir / f"{level_config['name']}_coverage_{safe_name}.png"
        print(f"\n  Mapping: {indicator}")
        plot_coverage_map(
            subset,
            boundary_features,
            indicator,
            output_path,
            level_config["data_col"],
            shapefile_id_col,
            background_boundary_features=bg_features,
        )

    print(f"  Saved {len(plot_indicators)} maps for {level_config['name']} level\n")

print("All administrative levels processed successfully!")
