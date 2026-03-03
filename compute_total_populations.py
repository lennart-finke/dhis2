#!/usr/bin/env python3
"""
Export region-level population estimates from WorldPop 100m x 100m grid data.
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
from tqdm import tqdm
import concurrent.futures

# Paths
import argparse

# Paths
WORLDPOP_DIR = Path("data/worldpop")
ADMIN_BOUNDS_GEOJSON = (
    "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin1.geojson"
)
ADMIN_BOUNDS_SHP = "data/admin_bounds/eth_admin_boundaries.shp/eth_admin1.shp"
OUTPUT_DIR = Path("estimates")
OUTPUT_DIR = Path("estimates")
# Default output filename (will be updated based on args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export region-level population estimates."
    )
    return parser.parse_args()


def process_year_data(
    worldpop_file,
    regions,
    zones,
    woredas,
    kebeles,
    regions_projected,
    zones_projected,
    woredas_projected,
    kebeles_projected,
):
    """
    Process a single WorldPop raster file for all administrative levels.
    """
    # Extract year from filename
    year = int(worldpop_file.stem.split("_")[2])

    results = []

    # Read worldpop raster path (rasterstats opens it internally)
    raster_path = str(worldpop_file)

    # --- Process Regions ---
    region_stats = zonal_stats(regions.geometry, raster_path, stats=["sum"])
    for idx, row in regions.iterrows():
        pop = region_stats[idx].get("sum")
        area_sq_km = regions_projected.iloc[idx].geometry.area / 1e6
        results.append(
            {
                "region": row["adm1_name"],
                "zone": "",
                "woreda": "",
                "kebele": "",
                "year": year,
                "population": int(pop) if pop else 0,
                "area_sq_km": area_sq_km,
            }
        )

    # --- Process Zones ---
    zone_stats = zonal_stats(zones.geometry, raster_path, stats=["sum"])
    for idx, row in zones.iterrows():
        pop = zone_stats[idx].get("sum")
        area_sq_km = zones_projected.iloc[idx].geometry.area / 1e6
        results.append(
            {
                "region": row["adm1_name"],
                "zone": row["adm2_name"],
                "woreda": "",
                "kebele": "",
                "year": year,
                "population": int(pop) if pop else 0,
                "area_sq_km": area_sq_km,
            }
        )

    # --- Process Woredas ---
    woreda_stats = zonal_stats(woredas.geometry, raster_path, stats=["sum"])
    for idx, row in woredas.iterrows():
        pop = woreda_stats[idx].get("sum")
        area_sq_km = woredas_projected.iloc[idx].geometry.area / 1e6
        results.append(
            {
                "region": row["adm1_name"],
                "zone": row["adm2_name"],
                "woreda": row["adm3_name"],
                "kebele": "",
                "year": year,
                "population": int(pop) if pop else 0,
                "area_sq_km": area_sq_km,
            }
        )

    # --- Process Kebeles ---
    if not kebeles.empty:
        kebele_stats = zonal_stats(kebeles.geometry, raster_path, stats=["sum"])
        for idx, row in kebeles.iterrows():
            pop = kebele_stats[idx].get("sum")
            area_sq_km = kebeles_projected.iloc[idx].geometry.area / 1e6
            results.append(
                {
                    "region": row["R_NAME"],
                    "zone": row["Z_NAME"],
                    "woreda": row["W_NAME"],
                    "kebele": row["RK_NAME"],
                    "year": year,
                    "population": int(pop) if pop else 0,
                    "area_sq_km": area_sq_km,
                }
            )

    return results


def main():
    parse_args()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Determine boundary files
    admin_bounds = ADMIN_BOUNDS_GEOJSON
    zones_filename = "eth_admin2.geojson"
    woredas_filename = "eth_admin3.geojson"
    kebeles_filename = "eth_admin4.geojson"

    # Set output filename based on source
    output_file = OUTPUT_DIR / "populations_worldpop_geojson.csv"

    # Read admin boundaries
    print("Loading boundaries...")
    regions = gpd.read_file(admin_bounds)
    print(f"Loaded {len(regions)} regions")

    zones_path = Path(admin_bounds).parent / zones_filename
    zones = gpd.read_file(zones_path)
    print(f"Loaded {len(zones)} zones")

    woredas_path = Path(admin_bounds).parent / woredas_filename
    woredas = gpd.read_file(woredas_path)
    print(f"Loaded {len(woredas)} woredas")

    kebeles_path = Path(admin_bounds).parent / kebeles_filename
    kebeles = gpd.read_file(kebeles_path)
    print(f"Loaded {len(kebeles)} kebeles")

    # Reproject to equal-area
    print("Reprojecting boundaries...")
    equal_area_crs = "EPSG:8857"
    regions_projected = regions.to_crs(equal_area_crs)
    zones_projected = zones.to_crs(equal_area_crs)
    woredas_projected = woredas.to_crs(equal_area_crs)
    kebeles_projected = kebeles.to_crs(equal_area_crs)

    # Find all worldpop files
    worldpop_files = sorted(WORLDPOP_DIR.glob("eth_pop_*_CN_100m_R2025A_v1.tif"))

    # Process in parallel
    print(f"Processing {len(worldpop_files)} years...")
    results = []

    for f in tqdm(worldpop_files, desc="Processing years"):
        year_results = process_year_data(
            f, regions, zones, woredas, kebeles,
            regions_projected, zones_projected,
            woredas_projected, kebeles_projected
        )
        results.extend(year_results)

    # Create dataframe in long format
    df = pd.DataFrame(results)

    assert not df.empty

    # Adjust for systematic underestimation of rural populations
    print("\nApplying low-density area adjustment...")

    # Calculate population density for all years
    df["density"] = df["population"] / df["area_sq_km"]

    # Use 2022 as the reference year to identify low-density areas
    df_2022 = df[df["year"] == 2022].copy()

    # Create a unique identifier for each area (region/zone/woreda/kebele combination)
    df["area_id"] = (
        df["region"] + "|" + df["zone"] + "|" + df["woreda"] + "|" + df["kebele"]
    )
    df_2022["area_id"] = (
        df_2022["region"]
        + "|"
        + df_2022["zone"]
        + "|"
        + df_2022["woreda"]
        + "|"
        + df_2022["kebele"]
    )

    # Identify which areas have density < 300 in 2022
    low_density_areas = set(df_2022[df_2022["density"] < 300]["area_id"])

    print(
        f"Identified {len(low_density_areas)} unique low-density areas based on 2022 data"
    )

    # Apply adjustment to ALL years for areas that are low-density in 2022
    low_density_mask = df["area_id"].isin(low_density_areas)
    num_adjusted_records = low_density_mask.sum()

    inflation_factor = 1 / (1 - 0.53)
    df.loc[low_density_mask, "population"] = (
        df.loc[low_density_mask, "population"] * inflation_factor
    ).astype(int)

    print(f"Applied adjustment to {num_adjusted_records} records across all years")

    # Recalculate density after adjustment
    df["density"] = df["population"] / df["area_sq_km"]

    # Drop the temporary area_id column
    df = df.drop(columns=["area_id"])

    # Apply census calibration to matched woredas
    print("\nApplying census calibration to matched woredas...")

    # Load GeoJSON→Census mapping (from match_woreda_names.py)
    geo_census_mapping = pd.read_csv("estimates/geojson_census_mapping.csv")

    # Load census data for 2022 projection
    census_df = pd.read_csv("data/census_populations/census_data.csv")
    census_woredas = census_df[
        census_df["status"].isin(["District", "Town", "Sub City"])
    ].copy()
    census_woredas = census_woredas[["name", "pop_projection_2022"]].rename(
        columns={"name": "census_woreda", "pop_projection_2022": "census_pop_2022"}
    )

    # Merge mapping with census population data
    matched = geo_census_mapping.merge(census_woredas, on="census_woreda", how="left")

    # Get 2022 GeoJSON estimates
    geojson_2022 = df[
        (df["woreda"].notna()) & (df["woreda"] != "") & (df["year"] == 2022)
    ][["woreda", "population"]].copy()

    # Merge to get both 2022 estimates
    matched = matched.merge(
        geojson_2022, left_on="geo_woreda", right_on="woreda", how="left"
    )

    # Filter to valid data
    matched = matched[
        matched["census_pop_2022"].notna()
        & matched["population"].notna()
        & (matched["population"] > 0)
    ].copy()

    print(
        f"Found {len(matched)} GeoJSON woredas matched to {matched['census_woreda'].nunique()} census woredas"
    )

    # Handle many-to-one mappings: group by census_woreda
    # For each census woreda, sum all GeoJSON populations that map to it
    grouped = (
        matched.groupby("census_woreda")
        .agg(
            {
                "population": "sum",  # Sum of all GeoJSON woredas mapping to this census woreda
                "census_pop_2022": "first",  # Census population (same for all in group)
                "geo_woreda": list,  # List of GeoJSON woreda names in this group
            }
        )
        .reset_index()
    )

    # Compute geometric mean and scaling factor for each group
    grouped["geometric_mean"] = (
        grouped["census_pop_2022"] * grouped["population"]
    ) ** 0.5
    grouped["scale_factor"] = grouped["census_pop_2022"] / grouped["geometric_mean"]

    print(
        f"Scale factors range: {grouped['scale_factor'].min():.3f} to {grouped['scale_factor'].max():.3f}"
    )
    print(f"Median scale factor: {grouped['scale_factor'].median():.3f}")

    # Create a mapping from woreda name to scale factor
    woreda_to_scale = {}
    for _, row in grouped.iterrows():
        scale_factor = row["scale_factor"]
        for woreda_name in row["geo_woreda"]:
            woreda_to_scale[woreda_name] = scale_factor

    # Apply scaling factors to all years for matched woredas
    calibrated_count = 0
    for woreda_name, scale_factor in woreda_to_scale.items():
        mask = df["woreda"] == woreda_name
        if mask.sum() > 0:
            df.loc[mask, "population"] = (
                df.loc[mask, "population"] * scale_factor
            ).astype(int)
            calibrated_count += 1

    print(f"Applied census calibration to {calibrated_count} woredas")

    # Show statistics on many-to-one mappings
    multi_match = grouped[grouped["geo_woreda"].apply(len) > 1]
    if len(multi_match) > 0:
        print(
            f"Note: {len(multi_match)} census woredas have multiple GeoJSON woredas mapped to them"
        )

    # Sort by region, zone, woreda, kebele, and year
    df = df.sort_values(["region", "zone", "woreda", "kebele", "year"])

    # Export to CSV
    df.to_csv(output_file, index=False)

    print(f"\nExported population estimates to {output_file}")
    print(
        f"Records: {len(df)} (Regions: {df['region'].nunique()}, Years: {df['year'].nunique()})"
    )
    print("\nPreview:")
    print(df.head(20))


if __name__ == "__main__":
    main()
