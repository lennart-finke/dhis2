"""
Compute woreda-level travel distances to health facilities.
Processes both walking and motorized travel time rasters.
Distance estimates are weighted by 2022 population data.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
from tqdm import tqdm

WALKING_RASTER = "data/road_access/service_area_Ethiopia_walking.tif"
MOTORIZED_RASTER = "data/road_access/service_area_Ethiopia_motorised.tif"
POPULATION_RASTER = "data/worldpop/eth_pop_2022_CN_100m_R2025A_v1.tif"
WOREDAS = "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin3.geojson"
OUTPUT = "estimates/woreda_travel_distances.csv"

# Load woreda boundaries
woredas = gpd.read_file(WOREDAS)

# Open all rasters
walking_src = rasterio.open(WALKING_RASTER)
motorized_src = rasterio.open(MOTORIZED_RASTER)
pop_src = rasterio.open(POPULATION_RASTER)

results = []

for idx, row in tqdm(woredas.iterrows(), total=len(woredas), desc="Processing woredas"):
    # Get the geometry for this woreda
    geom = [row.geometry.__geo_interface__]

    # Extract walking travel time data
    walking_data, walking_transform = mask(walking_src, geom, crop=True, nodata=-9999)
    walking_array = walking_data[0]

    # Extract motorized travel time data
    motorized_data, motorized_transform = mask(
        motorized_src, geom, crop=True, nodata=-9999
    )
    motorized_array = motorized_data[0]

    # Extract population data
    pop_data, pop_transform = mask(pop_src, geom, crop=True, nodata=-9999)
    pop_array = pop_data[0]

    # Ensure all arrays have the same shape
    min_height = min(
        walking_array.shape[0], motorized_array.shape[0], pop_array.shape[0]
    )
    min_width = min(
        walking_array.shape[1], motorized_array.shape[1], pop_array.shape[1]
    )
    walking_array = walking_array[:min_height, :min_width]
    motorized_array = motorized_array[:min_height, :min_width]
    pop_array = pop_array[:min_height, :min_width]

    # Create masks for valid data
    walking_valid = (
        (walking_array != -9999) & (~np.isnan(walking_array)) & (walking_array >= 0)
    )
    motorized_valid = (
        (motorized_array != -9999)
        & (~np.isnan(motorized_array))
        & (motorized_array >= 0)
    )
    pop_valid = (pop_array > 0) & (~np.isnan(pop_array))

    # Combined mask for walking: both walking and pop must be valid
    walking_mask = walking_valid & pop_valid

    # Combined mask for motorized: both motorized and pop must be valid
    motorized_mask = motorized_valid & pop_valid

    # Initialize result dictionary
    result = {
        "region": row["adm1_name"],
        "zone": row["adm2_name"],
        "woreda": row["adm3_name"],
        "pcode": row.get("adm3_pcode", ""),
    }

    # Process Walking Data
    assert walking_mask.any()
    walking_values = walking_array[walking_mask]
    pop_values_walking = pop_array[walking_mask]

    # Calculate weighted and unweighted means (in seconds)
    weighted_mean_walking = np.average(walking_values, weights=pop_values_walking)
    unweighted_mean_walking = np.mean(walking_values)
    min_walking = np.min(walking_values)
    max_walking = np.max(walking_values)

    # Convert to hours and add to result
    result.update(
        {
            "walking_weighted_mean_hrs": weighted_mean_walking / 3600,
            "walking_unweighted_mean_hrs": unweighted_mean_walking / 3600,
            "walking_min_hrs": min_walking / 3600,
            "walking_max_hrs": max_walking / 3600,
            "walking_population": float(np.sum(pop_values_walking)),
            "walking_pixel_count": int(walking_mask.sum()),
        }
    )

    # Process Motorized Data
    assert motorized_mask.any()
    motorized_values = motorized_array[motorized_mask]
    pop_values_motorized = pop_array[motorized_mask]

    # Calculate weighted and unweighted means (in seconds)
    weighted_mean_motorized = np.average(motorized_values, weights=pop_values_motorized)
    unweighted_mean_motorized = np.mean(motorized_values)
    min_motorized = np.min(motorized_values)
    max_motorized = np.max(motorized_values)

    # Convert to hours and add to result
    result.update(
        {
            "motorized_weighted_mean_hrs": weighted_mean_motorized / 3600,
            "motorized_unweighted_mean_hrs": unweighted_mean_motorized / 3600,
            "motorized_min_hrs": min_motorized / 3600,
            "motorized_max_hrs": max_motorized / 3600,
            "motorized_population": float(np.sum(pop_values_motorized)),
            "motorized_pixel_count": int(motorized_mask.sum()),
        }
    )

    results.append(result)


# Close rasters
walking_src.close()
motorized_src.close()
pop_src.close()

# Create dataframe and save
df = pd.DataFrame(results)
df.to_csv(OUTPUT, index=False)

print(f"Saved {len(df)} woredas to {OUTPUT}")
print("\nSummary statistics:")
print("\nWalking:")
print(
    df[
        [
            "walking_weighted_mean_hrs",
            "walking_unweighted_mean_hrs",
            "walking_population",
        ]
    ].describe()
)
print(
    f"\nWoredas with valid walking data: {df['walking_weighted_mean_hrs'].notna().sum()}"
)
print(f"Total population covered (walking): {df['walking_population'].sum():,.0f}")

print("\nMotorized:")
print(
    df[
        [
            "motorized_weighted_mean_hrs",
            "motorized_unweighted_mean_hrs",
            "motorized_population",
        ]
    ].describe()
)
print(
    f"\nWoredas with valid motorized data: {df['motorized_weighted_mean_hrs'].notna().sum()}"
)
print(f"Total population covered (motorized): {df['motorized_population'].sum():,.0f}")
