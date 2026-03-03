from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

DEFAULT_CACHE_DIR = Path("cache")
DEFAULT_ETHIOPIAN_YEAR_OFFSET = 8

# Map DHIS2 region names to WorldPop/population data region names
# Only exact equivalents are included; regions without matches are excluded
DHIS2_TO_WORLDPOP_REGIONS = {
    "Addis Ababa City Administration": "Addis Ababa",
    "Afar Region": "Afar",
    "Amhara Region": "Amhara",
    "Benishangul Gumuz Regional Health Bureau": "Benishangul Gumz",
    "Central Ethiopian region": "SNNP",  # Former SNNP region
    "Dire Dawa City Administration": "Dire Dawa",
    "Gambella Region": "Gambela",
    "Harari Region": "Harari",
    "Oromia Region": "Oromia",
    "SNNP Region": "SNNP",
    "Sidama Region": "Sidama",
    "Somali Region": "Somali",
    "South Ethiopia Region": "SNNP",  # Former SNNP region
    "South West Ethiopia Region": "South West Ethiopia",
    "South West Ethiopian Region": "South West Ethiopia",
    "Tigray Region": "Tigray",
    # Identity mappings (in case data is already normalized)
    "Addis Ababa": "Addis Ababa",
    "Afar": "Afar",
    "Amhara": "Amhara",
    "Benishangul Gumz": "Benishangul Gumz",
    "Dire Dawa": "Dire Dawa",
    "Gambela": "Gambela",
    "Harari": "Harari",
    "Oromia": "Oromia",
    "SNNP": "SNNP",
    "Sidama": "Sidama",
    "Somali": "Somali",
    "South West Ethiopia": "South West Ethiopia",
    "Tigray": "Tigray",
}

ETHIOPIAN_REGION_COLORS = {
    "Addis Ababa": "#009A44",  # Flag Green
    "Afar": "#FCDD09",  # Flag Yellow
    "Amhara": "#DA121A",  # Flag Red
    "Benishangul Gumuz": "#0F47AF",  # Emblem Blue
    "Central Ethiopia": "#8B4513",  # Saddle Brown (Earth)
    "Dire Dawa": "#FF6B35",  # Bright Coral/Orange
    "Gambela": "#20B2AA",  # Turquoise (Highlands water)
    "Harari": "#8B008B",  # Dark Magenta (Deep purple)
    "Oromia": "#CD7F32",  # Bronze/Copper
    "Sidama": "#2F4F4F",  # Dark Slate Grey/Charcoal
    "Somali": "#DEB887",  # Burlywood (Desert sand)
    "South Ethiopia": "#556B2F",  # Dark Olive Green
    "South West Ethiopia": "#8B0000",  # Dark Red/Maroon
    "Tigray": "#B8860B",  # Dark Goldenrod
    "SNNP": "#696969",  # Dim Grey (Neutral)
    "National": "#000000",  # Black
}

REGION_ALIASES = {
    # DHIS2 / Filename Variations
    "Addis Ababa City Administration": "Addis Ababa",
    "Afar Region": "Afar",
    "Amhara Region": "Amhara",
    "Benishangul Gumuz Regional Health Bureau": "Benishangul Gumuz",
    "Benishangul Gumz": "Benishangul Gumuz",  # Population data spelling
    "Dire Dawa City Administration": "Dire Dawa",
    "Gambella Region": "Gambela",
    "Harari Region": "Harari",
    "Oromia Region": "Oromia",
    "SNNP Region": "SNNP",
    "Sidama Region": "Sidama",
    "Somali Region": "Somali",
    "South West Ethiopia Region": "South West Ethiopia",
    "South West Ethiopian Region": "South West Ethiopia",
    "Tigray Region": "Tigray",
    # Specific filenames mappings
    "Gambella and Harari regions": "Gambela",  # Mapped to one color
    "Benishangul Gumuz and Central Ethiopia": "Benishangul Gumuz",
    "South-west Ethiopia": "South West Ethiopia",
}


def get_region_color(region_name: str) -> str:
    """
    Get the assigned color for a region.
    First resolves naming variations to a standard name.
    Raises KeyError if the region is not found in either aliases or standard colors.
    """
    # 1. Resolve alias if present, otherwise assume it's standard
    standard_name = REGION_ALIASES.get(region_name, region_name)

    # 2. Look up color (no fallback, will raise KeyError if missing)
    return ETHIOPIAN_REGION_COLORS[standard_name]


_ORGUNIT_LEVEL_RE = re.compile(r"^orgunitlevel(\d+)$")
_ETH_Q_RE = re.compile(r"^(\d{4})Q([1-4])$")


def normalize_dhis2_region_name(region_name: str) -> str:
    """
    Normalize DHIS2 region names to match WorldPop/population data region names.
    Only regions with exact matches in DHIS2_TO_WORLDPOP_REGIONS are mapped.
    """
    return DHIS2_TO_WORLDPOP_REGIONS.get(region_name, None)


def infer_orgunit_level_cols(columns: Iterable[str]) -> list[str]:
    """
    Return org-unit hierarchy columns sorted by level number, e.g.
    ['orgunitlevel1', 'orgunitlevel2', ...].
    """
    found: list[tuple[int, str]] = []
    for c in columns:
        m = _ORGUNIT_LEVEL_RE.match(c)
        if m:
            found.append((int(m.group(1)), c))
    return [c for _, c in sorted(found, key=lambda x: x[0])]


def drop_all_nan_columns(
    df: pd.DataFrame,
    *,
    keep: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Drop columns that are entirely NaN.

    Notes:
    - Uses pandas NaN semantics; empty strings are NOT treated as NaN.
    - Columns listed in `keep` will be preserved even if all-NaN.
    """
    keep_set = set(keep or [])
    non_all_nan = df.columns[df.notna().any(axis=0)]
    cols = list(
        dict.fromkeys(list(non_all_nan) + [c for c in df.columns if c in keep_set])
    )
    return df.loc[:, cols].copy()


def ethiopian_periodcode_to_gregorian_date(
    periodcode: str,
    *,
    year_offset: int = DEFAULT_ETHIOPIAN_YEAR_OFFSET,
) -> pd.Timestamp | pd.NaT:
    """
    Convert DHIS2 Ethiopian periodcode (currently supports 'YYYYQn') into a
    Gregorian quarter-start date (Timestamp).

    Example:
      '2017Q1' -> 2025-01-01 (with default offset=8)
    """
    if periodcode is None or (isinstance(periodcode, float) and pd.isna(periodcode)):  # type: ignore[redundant-expr]
        return pd.NaT
    s = str(periodcode).strip()
    m = _ETH_Q_RE.match(s)
    if not m:
        return pd.NaT
    eth_year, quarter = int(m.group(1)), int(m.group(2))
    greg_year = eth_year + year_offset
    month = (quarter - 1) * 3 + 1
    return pd.Timestamp(year=greg_year, month=month, day=1)


def add_gregorian_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with a Gregorian date column derived from 'periodcode'.
    Adds column 'gregorian_date' using the default Ethiopian year offset (8).
    """
    out = df.copy()
    out["gregorian_date"] = out["periodcode"].apply(
        lambda x: ethiopian_periodcode_to_gregorian_date(
            x, year_offset=DEFAULT_ETHIOPIAN_YEAR_OFFSET
        )
    )
    return out


def _norm_level_value(v: object) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    return s if s else None


def _row_to_path(row_vals: Sequence[object]) -> tuple[str, ...]:
    parts: list[str] = []
    for v in row_vals:
        nv = _norm_level_value(v)
        if nv is not None:
            parts.append(nv)
    return tuple(parts)


def compute_leaf_paths_from_unique_paths(
    paths: Iterable[tuple[str, ...]],
) -> set[tuple[str, ...]]:
    """
    Given an iterable of hierarchy paths, compute which paths are leaves.

    A path is an internal node if it is a proper prefix of any other path.
    Leaves are those that are not proper prefixes.
    """
    path_set: set[tuple[str, ...]] = set()
    internal: set[tuple[str, ...]] = set()

    for p in paths:
        if not p:
            continue
        path_set.add(p)

    for p in path_set:
        # Mark every proper prefix as internal
        for i in range(1, len(p)):
            internal.add(p[:i])

    return path_set - internal


def _cache_key_for_csv(csv_path: Path, *, level_cols: Sequence[str] | None) -> str:
    st = csv_path.stat()
    payload = {
        "path": str(csv_path.resolve()),
        "mtime_ns": st.st_mtime_ns,
        "size": st.st_size,
        "level_cols": list(level_cols) if level_cols else None,
        "v": 1,
    }
    b = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _leaf_cache_file(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"dhis2_leaf_paths_{cache_key}.json.gz"


def compute_leaf_paths_from_csv(
    csv_path: Path,
    *,
    level_cols: Sequence[str] | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> tuple[set[tuple[str, ...]], list[str]]:
    """
    Compute leaf orgunit paths by scanning the CSV using orgunitlevelN columns.
    Results are cached in `cache_dir`.
    """
    csv_path = Path(csv_path)

    if level_cols is None:
        header = pd.read_csv(csv_path, nrows=0)
        level_cols = infer_orgunit_level_cols(header.columns)
    level_cols_list = list(level_cols)
    if not level_cols_list:
        return set(), []

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _cache_key_for_csv(csv_path, level_cols=level_cols_list)
    cache_file = _leaf_cache_file(cache_dir, cache_key)

    if use_cache and cache_file.exists():
        with gzip.open(cache_file, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        leaf_paths = {tuple(p) for p in payload["leaf_paths"]}
        cached_level_cols = list(payload["level_cols"])
        return leaf_paths, cached_level_cols

    df_levels = pd.read_csv(
        csv_path,
        usecols=level_cols_list,
        dtype=str,
        keep_default_na=True,
    )
    unique_paths: set[tuple[str, ...]] = set()
    for vals in df_levels.itertuples(index=False, name=None):
        p = _row_to_path(vals)
        if p:
            unique_paths.add(p)

    leaf_paths = compute_leaf_paths_from_unique_paths(unique_paths)

    payload = {
        "level_cols": level_cols_list,
        "leaf_paths": [list(p) for p in sorted(leaf_paths)],
    }
    with gzip.open(cache_file, "wt", encoding="utf-8") as f:
        json.dump(payload, f)

    return leaf_paths, level_cols_list


def merge_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge legacy 2017 column names with their standardized equivalents.
    Adds values from old columns to new ones, then drops the old columns.
    """
    df = df.copy()

    merge_pairs = [
        (
            "_2017_Number of children <5 year screened and have moderate acute malnutrition",
            "NUT_Children <5 Years Screened with MAM",
        ),
        (
            "_2017_Number of children <5 year screened and have severe acute malnutrition",
            "NUT_Children <5 Years Screened with SAM",
        ),
    ]

    for old_col, new_col in merge_pairs:
        old_vals = pd.to_numeric(df[old_col], errors="coerce").fillna(0.0)
        new_vals = pd.to_numeric(df[new_col], errors="coerce").fillna(0.0)
        df[new_col] = old_vals + new_vals
        df = df.drop(columns=[old_col])
    return df


def enforce_local_contraints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Screened for malnutrition should be at least the sum of MAM and SAM
    total = "NUT_Children < 5 years Screened for Acute Malnutrition"
    mam = "NUT_Children <5 Years Screened with MAM"
    sam = "NUT_Children <5 Years Screened with SAM"

    child_cols = [c for c in [mam, sam] if c in df.columns]
    total_orig = pd.to_numeric(df[total], errors="coerce")
    for c in child_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    required_sum = df[child_cols].sum(axis=1)
    df[total] = total_orig.where(
        total_orig.isna() | (total_orig >= required_sum), required_sum
    )

    # Weighed should be at least LBW
    weighed = "NUT_Live Births Weighed"
    lbw = "NUT_Newborns < 2500 gm"

    weighed_orig = pd.to_numeric(df[weighed], errors="coerce")
    df[lbw] = pd.to_numeric(df[lbw], errors="coerce").fillna(0.0)
    df[weighed] = weighed_orig.where(
        weighed_orig.isna() | (weighed_orig >= df[lbw]), df[lbw]
    )

    return df


def deduplicate_non_facility_rows(
    df: pd.DataFrame,
    *,
    csv_path: Path | None = None,
    level_cols: Sequence[str] | None = None,
    leaf_paths: set[tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """
    Remove non-facility (aggregated) rows by keeping only orgunit paths that are leaves
    in the orgunit hierarchy tree defined by orgunitlevelN columns.

    Leaf computation can be expensive; pass `csv_path` to enable on-disk caching in `cache/`.
    In the common case, callers should just do:
        deduplicate_non_facility_rows(df, csv_path=Path(".../region.csv"))

    Params:
    - csv_path: source CSV path (recommended) used to compute/cached leaf paths.
    - level_cols: explicit orgunitlevel columns to use; if omitted, inferred from df.
    - leaf_paths: precomputed set of leaf paths (tuples). If provided, no cache/scan needed.
    """
    if level_cols is None:
        level_cols = infer_orgunit_level_cols(df.columns)
    level_cols_list = list(level_cols)
    if not level_cols_list:
        return df.copy()

    if leaf_paths is None:
        if csv_path is None:
            # Compute from df directly (no cache); caller can pass leaf_paths for reuse.
            unique_paths = {
                _row_to_path(vals)
                for vals in df[level_cols_list].itertuples(index=False, name=None)
            }
            unique_paths.discard(tuple())
            leaf_paths = compute_leaf_paths_from_unique_paths(unique_paths)
        else:
            leaf_paths, _ = compute_leaf_paths_from_csv(
                Path(csv_path),
                level_cols=level_cols_list,
                cache_dir=DEFAULT_CACHE_DIR,
                use_cache=True,
            )

    # Build row-wise mask
    def is_leaf_row(vals: Sequence[object]) -> bool:
        p = _row_to_path(vals)
        if not p:
            return False  # Drop unmapped rows
        return p in leaf_paths  # type: ignore[operator]

    mask = [
        is_leaf_row(vals)
        for vals in df[level_cols_list].itertuples(index=False, name=None)
    ]
    return df.loc[mask].copy()


def prepare_dhis2(
    df: pd.DataFrame,
    *,
    csv_path: Path | None = None,
    level_cols: Sequence[str] | None = None,
    facilities_only: bool = False,
) -> pd.DataFrame:
    # Step 1: Deduplicate non-facility rows
    df = deduplicate_non_facility_rows(
        df,
        csv_path=csv_path,
        level_cols=level_cols,
    )

    # Step 1b: Drop remaining known administrative aggregates that masquerade as leaves
    if facilities_only and "organisationunitname" in df.columns:
        woreda_keywords = (" woreda", " health office", " zone", " region", " town", " sub city", " department")
        mask = ~df["organisationunitname"].str.lower().str.endswith(woreda_keywords, na=False)
        df = df[mask].copy()

    # Step 2: Merge duplicate columns
    df = merge_duplicate_columns(df)

    # Step 3: Enforce local constraints
    df = enforce_local_contraints(df)

    # Step 4: Add Gregorian date column
    df = add_gregorian_date_column(df)

    return df


# Map GeoJSON region names to normalized DHIS2 region names
GEO_TO_NORMALIZED_REGION = {
    "SNNP": "SNNP",
    "South West Ethiopia": "South West Ethiopia",
    "Sidama": "Sidama",
    "Addis Ababa": "Addis Ababa",
    "Addis Abeba": "Addis Ababa",
    "Afar": "Afar",
    "Amhara": "Amhara",
    "Amahara": "Amhara",
    "Benishangul Gumz": "Benishangul Gumz",
    "Benishangul Gumuz": "Benishangul Gumz",
    "Dire Dawa": "Dire Dawa",
    "Dredewa": "Dire Dawa",
    "Gambela": "Gambela",
    "Gambella": "Gambela",
    "Harari": "Harari",
    "Oromia": "Oromia",
    "OROMIYA": "Oromia",
    "Oromiya": "Oromia",
    "Somali": "Somali",
    "SOMALE KILLIL": "Somali",
    "Tigray": "Tigray",
    # DHIS2-style region names (from orgunitlevel2)
    "Addis Ababa City Administration": "Addis Ababa",
    "Afar Region": "Afar",
    "Amhara Region": "Amhara",
    "Benishangul Gumuz Regional Health Bureau": "Benishangul Gumz",
    "Central Ethiopian region": "SNNP",
    "Dire Dawa City Administration": "Dire Dawa",
    "Gambella Region": "Gambela",
    "Harari Region": "Harari",
    "Oromia Region": "Oromia",
    "SNNP Region": "SNNP",
    "Sidama Region": "Sidama",
    "Somali Region": "Somali",
    "South Ethiopia Region": "SNNP",
    "South West Ethiopia Region": "South West Ethiopia",
    "South West Ethiopian Region": "South West Ethiopia",
    "Tigray Region": "Tigray",
}


def load_zone_mapping() -> pd.DataFrame:
    """
    Load zone three-way mapping and normalize region names.
    Returns DataFrame with columns: region_norm, zone_dhis2, zone_geojson
    """
    mapping_path = Path("estimates/zone_three_way_mapping.csv")
    if not mapping_path.exists():
        return pd.DataFrame(columns=["region_norm", "zone_dhis2", "zone_geojson"])

    zone_mapping = pd.read_csv(mapping_path)
    zone_mapping["region_norm"] = zone_mapping["region"].apply(
        normalize_dhis2_region_name
    )
    zone_mapping = zone_mapping.drop_duplicates(subset=["region_norm", "zone_dhis2"])
    return zone_mapping


def load_woreda_mapping() -> pd.DataFrame:
    """
    Load woreda three-way mapping and normalize region names.
    Returns DataFrame with columns: region_norm, zone, woreda_dhis2, woreda_geojson, woreda_geojson_pcode
    """
    mapping_path = Path("estimates/woreda_three_way_mapping.csv")
    if not mapping_path.exists():
        return pd.DataFrame(
            columns=[
                "region_norm",
                "zone",
                "woreda_dhis2",
                "woreda_geojson",
                "woreda_geojson_pcode",
            ]
        )

    woreda_mapping = pd.read_csv(mapping_path)
    woreda_mapping["region_norm"] = woreda_mapping["region"].apply(
        normalize_dhis2_region_name
    )
    return woreda_mapping


def load_population_data(
    pop_file: str = "estimates/populations_worldpop_geojson.csv",
) -> pd.DataFrame:
    """
    Load population data and normalize region names.
    Returns DataFrame with columns: region, zone, woreda, year, population, etc.
    """
    pop_df = pd.read_csv(pop_file, low_memory=False)
    # Normalize GeoJSON region names to match DHIS2 normalized names
    pop_df["region"] = (
        pop_df["region"].map(GEO_TO_NORMALIZED_REGION).fillna(pop_df["region"])
    )
    return pop_df


def get_regional_population(pop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract regional-level population (rows where zone is empty).
    Returns DataFrame with columns: region, year, population
    """
    regional = pop_df[pop_df["zone"].isna() | (pop_df["zone"] == "")].copy()
    regional = regional.drop(columns=["zone", "woreda"], errors="ignore")
    return regional[["region", "year", "population"]].drop_duplicates()


def get_zonal_population(pop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract zonal-level population (rows where zone is set but woreda is empty).
    Also promotes woredas to zones for City Administrations (Addis Ababa, Harari).
    Returns DataFrame with columns: region, zone, year, population
    """
    # Standard zones: rows with zone but no woreda
    zonal = pop_df[
        (pop_df["zone"].notna())
        & (pop_df["zone"] != "")
        & (pop_df["woreda"].isna() | (pop_df["woreda"] == ""))
    ].copy()

    # City Admins: Promote Woredas to Zones
    # For Addis Ababa and Harari, the population file has the sub-cities/zones listed as "woredas" (Admin 3)
    # e.g. Addis Ababa -> Region 14 -> Bole (Bole is listed as woreda but is the zone/subcity we want)
    city_regions = ["Addis Ababa", "Harari"]
    city_woredas = pop_df[
        (pop_df["region"].isin(city_regions))
        & (pop_df["woreda"].notna())
        & (pop_df["woreda"] != "")
    ].copy()

    if not city_woredas.empty:
        # Use 'woreda' as 'zone' for these regions
        city_woredas["zone"] = city_woredas["woreda"]
        # Remove these regions from the standard zonal set to avoid duplicates/using the dummy "Region 14" zone
        zonal = zonal[~zonal["region"].isin(city_regions)]
        zonal = pd.concat([zonal, city_woredas], ignore_index=True)

    zonal = zonal.drop(columns=["woreda"], errors="ignore")
    return zonal[["region", "zone", "year", "population"]].drop_duplicates()


def get_woreda_population(pop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract woreda-level population (rows where woreda is set but kebele is not).
    Returns DataFrame with columns: region, zone, woreda, year, population
    """
    mask = (pop_df["woreda"].notna()) & (pop_df["woreda"] != "")
    if "kebele" in pop_df.columns:
        mask = mask & ((pop_df["kebele"].isna()) | (pop_df["kebele"] == ""))

    woreda = pop_df[mask].copy()
    return woreda[["region", "zone", "woreda", "year", "population"]].drop_duplicates()


def get_kebele_population(pop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract kebele-level population (rows where kebele is set).
    Returns DataFrame with columns: region, zone, woreda, kebele, year, population
    """
    if "kebele" not in pop_df.columns:
        return pd.DataFrame(
            columns=["region", "zone", "woreda", "kebele", "year", "population"]
        )

    kebele = pop_df[(pop_df["kebele"].notna()) & (pop_df["kebele"] != "")].copy()
    return kebele[
        ["region", "zone", "woreda", "kebele", "year", "population"]
    ].drop_duplicates()


def map_dhis2_zones_to_geojson(
    dhis_df: pd.DataFrame,
    zone_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map DHIS2 zone names to GeoJSON zone names.

    Args:
        dhis_df: DataFrame with 'region' (normalized) and 'zone' columns
        zone_mapping: Zone mapping from load_zone_mapping()

    Returns:
        DataFrame with added 'zone_geojson' column
    """
    if zone_mapping.empty:
        dhis_df = dhis_df.copy()
        dhis_df["zone_geojson"] = None
        return dhis_df

    result = dhis_df.merge(
        zone_mapping[["region_norm", "zone_dhis2", "zone_geojson"]],
        left_on=["region", "zone"],
        right_on=["region_norm", "zone_dhis2"],
        how="left",
    )
    result = result.drop(columns=["region_norm", "zone_dhis2"], errors="ignore")
    return result


def map_dhis2_woredas_to_geojson(
    dhis_df: pd.DataFrame,
    woreda_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map DHIS2 woreda names to GeoJSON woreda names.

    Args:
        dhis_df: DataFrame with 'region' (normalized), 'zone', and 'woreda' columns
        woreda_mapping: Woreda mapping from load_woreda_mapping()

    Returns:
        DataFrame with added 'woreda_geojson' column
    """
    assert not woreda_mapping.empty, "Woreda mapping is empty"

    # Prepare mapping subset
    mapping_subset = woreda_mapping[
        ["region_norm", "zone", "woreda_dhis2", "woreda_geojson"]
    ].drop_duplicates()

    result = dhis_df.merge(
        mapping_subset,
        left_on=["region", "zone", "woreda"],
        right_on=["region_norm", "zone", "woreda_dhis2"],
        how="left",
    )
    result = result.drop(columns=["region_norm", "woreda_dhis2"], errors="ignore")
    return result
