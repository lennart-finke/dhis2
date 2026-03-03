import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import linregress
import multiprocessing as mp

from dhis2_utils import prepare_dhis2


def process_facility(args):
    """
    Process a single facility's data for outlier detection using an OLS linear fit.
    """
    facility, facility_df, numeric_cols = args
    facility_outliers = []

    # Needs to be sorted chronologically for proper trend fitting
    facility_df = facility_df.sort_values("periodid")

    for col in numeric_cols:
        # Get non-zero, non-NaN values
        values = pd.to_numeric(facility_df[col], errors="coerce")
        mask = (values != 0) & values.notna()
        non_zero_df = facility_df[mask]

        # We need at least 3 points to do meaningful OLS and standard deviation of residuals
        if len(non_zero_df) < 3:
            continue

        y = pd.to_numeric(non_zero_df[col]).values

        # If gregorian_date is available from prepare_dhis2, use it as 'x' (days since earliest record)
        # Else, fall back to relative order.
        if (
            "gregorian_date" in non_zero_df.columns
            and not non_zero_df["gregorian_date"].isna().all()
        ):
            x = (
                non_zero_df["gregorian_date"] - non_zero_df["gregorian_date"].min()
            ).dt.days.values
        else:
            x = np.arange(len(y))

        # Check standard deviation of x to avoid ZeroDivisionError in linregress
        if x.std() == 0:
            continue

        slope, intercept, _, _, _ = linregress(x, y)
        predicted = intercept + slope * x

        # Predicted value shouldn't be negative in this context
        predicted = np.maximum(predicted, 0)

        residuals = y - predicted
        std = np.std(residuals)

        if std == 0:
            continue

        # The threshold replaces mean with OLS smoothed prediction
        threshold = np.minimum(5 * predicted, predicted + 2 * std)

        # Find values exceeding threshold
        outlier_mask = y > threshold
        if np.any(outlier_mask):
            outlier_rows = non_zero_df[outlier_mask]
            outlier_y = y[outlier_mask]
            outlier_pred = predicted[outlier_mask]

            for idx, (original_idx, row) in enumerate(outlier_rows.iterrows()):
                facility_outliers.append(
                    {
                        "facility": facility,
                        "column": col,
                        "periodid": row["periodid"],
                        "value": outlier_y[idx],
                        "predicted": outlier_pred[idx],
                        "std": std,
                        "threshold": threshold[outlier_mask][idx],
                        "std_above": (outlier_y[idx] - outlier_pred[idx]) / std
                        if std > 0
                        else 0,
                    }
                )
    return facility_outliers


def main():
    BASE_DIR = Path("data/dhis2/25_12_12")
    cache_file = Path("cache/outliers_ols.csv")

    if not BASE_DIR.exists():
        print(f"Error: Directory {BASE_DIR} does not exist.")
        return

    print("Loading data...")
    dhis2 = []

    csv_paths = list(sorted(BASE_DIR.glob("*.csv")))
    for csv_path in tqdm(csv_paths, desc="Reading CSVs"):
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=True)
        df = prepare_dhis2(df, csv_path=csv_path)
        dhis2.append(df)

    dhis2 = pd.concat(dhis2, ignore_index=True)

    # Get numeric columns (exclude metadata columns)
    exclude_cols = {
        "organisationunitid",
        "organisationunitname",
        "organisationunitcode",
        "organisationunitdescription",
        "periodid",
        "periodname",
        "periodcode",
        "perioddescription",
        "gregorian_date",
        "year",
    }
    # Add all orgunitlevel columns
    for col in dhis2.columns:
        if col.startswith("orgunitlevel"):
            exclude_cols.add(col)

    print("Identifying numeric columns...")
    numeric_cols = []
    for col in dhis2.columns:
        if col in exclude_cols:
            continue
        try:
            # Check if column is numeric. At least some non-null values must be parseable.
            converted = pd.to_numeric(dhis2[col].dropna().iloc[:100], errors="coerce")
            if not converted.isna().all():
                numeric_cols.append(col)
        except Exception:
            continue

    print(f"Found {len(numeric_cols)} numeric columns.")

    print("Preparing tasks for multiprocessing...")
    facilities = dhis2["organisationunitname"].unique()

    # Grouping dataframe by facility
    grouped = dhis2.groupby("organisationunitname")
    tasks = [
        (facility, grouped.get_group(facility), numeric_cols) for facility in facilities
    ]

    print(f"Processing {len(tasks)} facilities using multiprocessing...")
    outliers = []

    # Use multiprocessing to detect outliers
    with mp.Pool(mp.cpu_count()) as pool:
        for result in tqdm(
            pool.imap_unordered(process_facility, tasks),
            total=len(tasks),
            desc="Detecting Outliers",
        ):
            outliers.extend(result)

    outliers_df = pd.DataFrame(outliers)
    print(f"Found {len(outliers_df)} outliers")

    # Save outliers to cache
    Path("cache").mkdir(exist_ok=True)
    outliers_df.to_csv(cache_file, index=False)
    print(f"Saved raw outliers to {cache_file}")

    if not outliers_df.empty:
        outliers_3ds = outliers_df[outliers_df["std_above"] > 3].copy()

        print("\nCalculating outlier percentages by region...")
        # Map facility to region
        dhis2_facilities = (
            dhis2.groupby("organisationunitname")["orgunitlevel2"].first().to_dict()
        )
        outliers_3ds["region"] = outliers_3ds["facility"].map(dhis2_facilities)

        total_per_region = pd.Series(dhis2_facilities).value_counts()

        # Calculate cross tabulation of unique facilities with outliers
        counts = outliers_3ds.pivot_table(
            index="region",
            columns="column",
            values="facility",
            aggfunc="nunique",
            margins=True,
        )

        # Align denominators with counts index and handle the 'All' margin
        denominators = total_per_region.reindex(counts.index)
        if "All" in counts.index:
            denominators.loc["All"] = total_per_region.sum()

        # Perform row-wise division
        percentage_outliers = (
            counts.div(denominators, axis=0).mul(100).round(2).fillna(0)
        )

        percentage_outliers.to_csv("cache/percentage_outliers_ols.csv")
        print("Saved summary table to cache/percentage_outliers_ols.csv")

        print(percentage_outliers.to_string())


if __name__ == "__main__":
    main()
