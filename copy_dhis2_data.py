import os
import glob
import pandas as pd
import numpy as np
import multiprocessing as mp
from scipy.stats import linregress
from tqdm import tqdm

from dhis2_utils import add_gregorian_date_column

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
    "_2017_Number of children <5 year screened and have moderate acute malnutrition",
    "_2017_Number of children <5 year screened and have severe acute malnutrition",
]

METADATA_COLS = {
    "organisationunitid",
    "organisationunitname",
    "organisationunitcode",
    "organisationunitdescription",
    "orgunitlevel2",
    "orgunitlevel3",
    "orgunitlevel4",
    "orgunitlevel5",
    "periodid",
    "periodname",
    "perioddescription",
    "periodcode",
    "gregorian_date",
    "year",
}

SRC_DIR = "data/dhis2/25_12_12_raw"
DST_DIR = "data/dhis2/25_12_12"


def process_facility(args):
    facility, facility_df, numeric_cols = args

    replaced_count = 0

    # Needs to be sorted chronologically for proper trend fitting
    if "periodid" in facility_df.columns:
        facility_df = facility_df.sort_values("periodid").copy()
    else:
        facility_df = facility_df.copy()

    facility_df = facility_df.reset_index(drop=True)

    for col in numeric_cols:
        if col not in facility_df.columns:
            continue

        values = pd.to_numeric(facility_df[col], errors="coerce")
        mask = (values != 0) & values.notna()
        non_zero_df = facility_df[mask]

        # We need at least 3 points to do meaningful OLS and standard deviation of residuals
        if len(non_zero_df) < 3:
            continue

        y = pd.to_numeric(non_zero_df[col]).values

        if (
            "gregorian_date" in non_zero_df.columns
            and not non_zero_df["gregorian_date"].isna().all()
        ):
            x = (
                non_zero_df["gregorian_date"] - non_zero_df["gregorian_date"].min()
            ).dt.days.values
        else:
            x = np.arange(len(y))

        if x.std() == 0:
            continue

        slope, intercept, _, _, _ = linregress(x, y)
        predicted = intercept + slope * x

        predicted = np.maximum(predicted, 0)

        residuals = y - predicted
        std = np.std(residuals)

        if std == 0:
            continue

        threshold = np.minimum(5 * predicted, predicted + 2 * std)

        outlier_mask = y > threshold

        std_above = np.zeros_like(y, dtype=float)
        std_above[outlier_mask] = (y[outlier_mask] - predicted[outlier_mask]) / std

        # 5 std threshold
        final_outlier_mask = std_above > 5

        if np.any(final_outlier_mask):
            outlier_indices = non_zero_df.index[final_outlier_mask]

            col_values = values

            for idx in outlier_indices:
                pos = facility_df.index.get_loc(idx)

                prev_val = np.nan
                next_val = np.nan

                if pos > 0:
                    prev_val = col_values.iloc[pos - 1]
                if pos < len(facility_df) - 1:
                    next_val = col_values.iloc[pos + 1]

                neighbors = [v for v in (prev_val, next_val) if pd.notna(v)]
                if neighbors:
                    avg_val = np.mean(neighbors)
                    facility_df.at[idx, col] = avg_val
                    replaced_count += 1
                else:
                    facility_df.at[idx, col] = np.nan
                    replaced_count += 1

    return facility_df, replaced_count


def main():
    assert os.path.exists(SRC_DIR)
    os.makedirs(DST_DIR, exist_ok=True)

    csv_files = glob.glob(os.path.join(SRC_DIR, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {SRC_DIR}")
        return

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")

        df = pd.read_csv(file_path, low_memory=False, dtype=str, keep_default_na=True)

        df = add_gregorian_date_column(df)
        df["year"] = df["gregorian_date"].dt.year

        requested_cols = set(INDICATORS).union(METADATA_COLS)
        present_cols = [c for c in df.columns if c in requested_cols]

        missing_cols = requested_cols - set(df.columns)
        if missing_cols:
            really_missing = [
                c
                for c in missing_cols
                if c not in ["gregorian_date", "year"] or ("periodcode" in df.columns)
            ]
            if really_missing:
                pass  # omit warning to avoid cluttering tqdm

        df_filtered = df[present_cols]

        # Identify numeric columns for this file
        numeric_cols = [c for c in present_cols if c in INDICATORS]
        assert numeric_cols
        facilities = df_filtered["organisationunitname"].unique()
        grouped = df_filtered.groupby("organisationunitname")
        tasks = [
            (facility, grouped.get_group(facility), numeric_cols)
            for facility in facilities
        ]

        print(f"  Detecting and replacing outliers across {len(tasks)} facilities...")
        processed_dfs = []
        total_replaced = 0
        with mp.Pool(mp.cpu_count()) as pool:
            for facility_df, replaced in tqdm(
                pool.imap_unordered(process_facility, tasks),
                total=len(tasks),
                desc="  Facilities",
            ):
                processed_dfs.append(facility_df)
                total_replaced += replaced

        print(f"  Replaced {total_replaced} outliers.")
        if processed_dfs:
            df_filtered = pd.concat(processed_dfs, ignore_index=True)

        dst_path = os.path.join(DST_DIR, filename)
        df_filtered.to_csv(dst_path, index=False)
        print(f"  Saved to {dst_path}")


if __name__ == "__main__":
    main()
