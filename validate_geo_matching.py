import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Point
import math

# Config paths
ADMIN2_FILE = "data/admin_bounds/eth_admin_boundaries.geojson/eth_admin2.geojson"
FACILITY_BOUNDARY_MAPPING = "estimates/facility_boundary_mapping.csv"

# DHS Config
DHS_GPS_FILES = {
    2016: "data/dhs/ET_2016_DHS_11092025_2313_234201/ETGE71FL/ETGE71FL.shp",
    2019: "data/dhs/ET_2019_INTERIMDHS_11092025_2313_234201/ETGE71FL/ETGE71FL.shp",
}

# PMA Config
PMA_GPS_FILES = {
    2019: "data/pma/gps/PMA_ET_GPS_v1_01Nov2020.csv",
    2021: "data/pma/gps/PMAET_2021_GPS_v1_25Jul2025.csv",
    2023: "data/pma/gps/PMAET_2023_GPS_v1_14Aug2025.csv",
}


def load_admin2():
    return gpd.read_file(ADMIN2_FILE)


def load_facilities():
    if not Path(FACILITY_BOUNDARY_MAPPING).exists():
        print(f"Warning: {FACILITY_BOUNDARY_MAPPING} not found.")
        return gpd.GeoDataFrame()

    df = pd.read_csv(FACILITY_BOUNDARY_MAPPING, low_memory=False)
    # Filter for valid matched zone and coordinates
    # We use boundary_zone as the definition of 'matched'
    df = df[
        df["boundary_zone"].notna() & df["latitude"].notna() & df["longitude"].notna()
    ].copy()

    # Filter out facilities with invalid GPS coordinates (out of WGS84 range).
    # Some rows have lat/lon stored in unusual units (e.g. decimal seconds) which
    # causes geopandas to compute an infinite aspect ratio and crash during plot().
    valid_coords = df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)
    n_invalid = (~valid_coords).sum()
    if n_invalid:
        print(
            f"  Warning: dropping {n_invalid} facilities with invalid GPS coordinates."
        )
    df = df[valid_coords].copy()

    # Create Geometry
    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def load_dhs_clusters():
    dfs = []
    for year, path in DHS_GPS_FILES.items():
        if not Path(path).exists():
            print(f"Warning: DHS file {path} not found.")
            continue
        gdf = gpd.read_file(path)
        # Filter invalid coordinates (lat/long = 0 often indicates invalid/missing in DHS)
        gdf = gdf[(gdf["LATNUM"] != 0) & (gdf["LONGNUM"] != 0)].copy()
        gdf["year"] = year
        gdf["survey_type"] = "DHS"
        dfs.append(gdf)

    if not dfs:
        return gpd.GeoDataFrame(columns=["geometry"])
    return pd.concat(dfs, ignore_index=True)


def load_pma_clusters():
    dfs = []
    for year, path in PMA_GPS_FILES.items():
        if not Path(path).exists():
            print(f"Warning: PMA file {path} not found.")
            continue
        df = pd.read_csv(path)
        geometry = [Point(xy) for xy in zip(df.GPSLONG, df.GPSLAT)]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        gdf["year"] = year
        gdf["survey_type"] = "PMA"
        dfs.append(gdf)

    if not dfs:
        return gpd.GeoDataFrame(columns=["geometry"])
    return pd.concat(dfs, ignore_index=True)


def main():
    out_dir = Path("figures/gps/zone_validation_maps")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Admin Boundaries...")
    admin2 = load_admin2()

    print("Loading Facilities...")
    facilities = load_facilities()

    print("Loading DHS Clusters...")
    dhs = load_dhs_clusters()

    print("Loading PMA Clusters...")
    pma = load_pma_clusters()

    # Ensure uniform CRS
    target_crs = admin2.crs
    if not facilities.empty and facilities.crs != target_crs:
        facilities = facilities.to_crs(target_crs)
    if not dhs.empty and dhs.crs != target_crs:
        dhs = dhs.to_crs(target_crs)
    if not pma.empty and pma.crs != target_crs:
        pma = pma.to_crs(target_crs)

    # Spatially join surveys to zones (admin2) to assign zone membership
    # We use sjoin for surveys as their association is purely spatial
    print("Spatially joining surveys to zones...")
    dhs_joined = (
        gpd.sjoin(
            dhs, admin2[["adm2_name", "geometry"]], how="inner", predicate="within"
        )
        if not dhs.empty
        else dhs
    )
    pma_joined = (
        gpd.sjoin(
            pma, admin2[["adm2_name", "geometry"]], how="inner", predicate="within"
        )
        if not pma.empty
        else pma
    )

    unique_zones = sorted(admin2["adm2_name"].unique())
    print(f"Found {len(unique_zones)} zones. generating maps...")

    for i, zone_name in enumerate(unique_zones):
        if pd.isna(zone_name):
            continue

        print(f"  [{i+1}/{len(unique_zones)}] Plotting {zone_name}...")

        # Filter Data
        zone_geom = admin2[admin2["adm2_name"] == zone_name]

        # Facilities: use the matched boundary_zone
        zone_facilities = (
            facilities[facilities["boundary_zone"] == zone_name]
            if not facilities.empty
            else facilities
        )

        # Surveys: use matched spatial join
        zone_dhs = (
            dhs_joined[dhs_joined["adm2_name"] == zone_name]
            if not dhs_joined.empty
            else dhs_joined
        )
        zone_pma = (
            pma_joined[pma_joined["adm2_name"] == zone_name]
            if not pma_joined.empty
            else pma_joined
        )

        # Setup Plot
        fig, ax = plt.subplots(figsize=(10, 10))

        # 1. Plot Zone Boundary
        zone_geom.plot(
            ax=ax, facecolor="whitesmoke", edgecolor="black", linewidth=1.5, zorder=1
        )

        # 2. Plot Facilities
        if not zone_facilities.empty:
            zone_facilities.plot(
                ax=ax,
                color="blue",
                marker="o",
                markersize=15,
                alpha=0.5,
                label=f"Facilities (n={len(zone_facilities)})",
                zorder=2,
            )

        # 3. Plot DHS
        if not zone_dhs.empty:
            zone_dhs.plot(
                ax=ax,
                color="red",
                marker="^",
                markersize=25,
                alpha=0.7,
                label=f"DHS Clusters (n={len(zone_dhs)})",
                zorder=3,
            )

        # 4. Plot PMA
        if not zone_pma.empty:
            zone_pma.plot(
                ax=ax,
                color="green",
                marker="s",
                markersize=25,
                alpha=0.7,
                label=f"PMA EAs (n={len(zone_pma)})",
                zorder=4,
            )

        # Set bounds with some padding
        if not zone_geom.empty:
            minx, miny, maxx, maxy = zone_geom.total_bounds
            pad_x = (maxx - minx) * 0.1
            pad_y = (maxy - miny) * 0.1
            ax.set_xlim(minx - pad_x, maxx + pad_x)
            ax.set_ylim(miny - pad_y, maxy + pad_y)

            # Add 10km Scale Bar
            # Calculate scale length in degrees
            # 1 deg lat approx 111.32 km
            # 1 deg lon approx 111.32 * cos(lat) km
            center_lat = (miny + maxy) / 2
            lat_rad = math.radians(center_lat)
            km_per_deg_lon = 111.32 * math.cos(lat_rad)

            # Check if we are close to pole or something weird (unlikely in Ethiopia)
            if km_per_deg_lon < 0.1:
                km_per_deg_lon = 1.0  # fallback

            scale_len_degrees = 10.0 / km_per_deg_lon

            # Position in bottom left with some padding relative to the plot area
            # (minx - pad_x) is the left edge of the visible plot
            # (miny - pad_y) is the bottom edge of the visible plot

            # Let's put it slightly offset from bottom left corner
            # Offset by 5% of the total visible width/height
            visible_width = (maxx + pad_x) - (minx - pad_x)
            visible_height = (maxy + pad_y) - (miny - pad_y)

            sb_x = (minx - pad_x) + visible_width * 0.05
            sb_y = (miny - pad_y) + visible_height * 0.05

            # Draw black line
            ax.plot(
                [sb_x, sb_x + scale_len_degrees],
                [sb_y, sb_y],
                color="black",
                linewidth=3,
                zorder=10,
            )

            # Draw ends (ticks)
            tick_height = visible_height * 0.01
            ax.plot(
                [sb_x, sb_x],
                [sb_y - tick_height / 2, sb_y + tick_height / 2],
                color="black",
                linewidth=3,
                zorder=10,
            )
            ax.plot(
                [sb_x + scale_len_degrees, sb_x + scale_len_degrees],
                [sb_y - tick_height / 2, sb_y + tick_height / 2],
                color="black",
                linewidth=3,
                zorder=10,
            )

            # Add text
            ax.text(
                sb_x + scale_len_degrees / 2,
                sb_y + tick_height,
                "10 km",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                zorder=10,
            )

        ax.set_title(f"{zone_name}")
        ax.set_axis_off()
        plt.legend(loc="upper right")

        # Save
        safe_name = str(zone_name).replace("/", "_").replace(" ", "_")
        out_path = out_dir / f"{safe_name}.pdf"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

    print(f"Maps saved to {out_dir}")


if __name__ == "__main__":
    main()
