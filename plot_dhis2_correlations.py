import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from dhis2_utils import (
    prepare_dhis2,
    ETHIOPIAN_REGION_COLORS,
    normalize_dhis2_region_name,
)

# Full list of indicators
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
]

# Short names for plotting
INDICATOR_NAMES = {
    "EPI_<1 Year Received 3rd Dose Penta": "Penta 3",
    "EPI_Children <1 Year Measels- 1st Dose": "Measles",
    "NUT_Children 6-59 Months received Vitamin A by Age": "Vitamin A",
    "NUT_Children 6-59 Months Received Vitamin A by Dose": "Vitamin A",
    "NUT_Children 24 - 59 Months Dewormed": "Dewormed",
    "Number of pregnant women who received ANC first visit by gestational week": "ANC 1",
    "MAT_Skilled Birth Attendance": "SBA",
    "MAT_Births attended by Level IV HEW And Nurses at HP": "HEW Births",
    "NUT_Children < 5 years Screened for Acute Malnutrition": "Screened Acute",
    "NUT_Children <5 Years Screened with MAM": "Screened MAM",
    "NUT_Children <5 Years Screened with SAM": "Screened SAM",
    "NUT_Live Births Weighed": "Weighed",
    "NUT_Newborns < 2500 gm": "Low Birth Weight",
    "MAT_Births By Caesarean Section": "C-Section",
}

BASE_DIR = Path("data/dhis2/25_12_12")
FACILITY_MAPPING = Path("estimates/facility_boundary_mapping.csv")


def load_data():
    # Load facility mapping for regions
    print("Loading facility boundary mapping...")
    facility_map = pd.read_csv(FACILITY_MAPPING)
    # Mapping uses dhis2_id, we need a map from id to region
    # Ensure no duplicates for ID
    facility_map = facility_map.drop_duplicates(subset="dhis2_id")
    id_to_region = facility_map.set_index("dhis2_id")["boundary_region"].to_dict()

    all_data = []
    print("Loading DHIS2 data...")
    for csv_path in sorted(BASE_DIR.glob("*.csv")):
        # Read raw
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=True)

        # Prepare using utils (deduplication, date parsing, etc.)
        df = prepare_dhis2(df, csv_path=csv_path, facilities_only=True)

        # Filter columns
        available_indicators = [c for c in df.columns if c in INDICATORS]

        if not available_indicators:
            continue

        keep_cols = ["organisationunitid", "organisationunitname", "gregorian_date"] + available_indicators
        df_subset = df[keep_cols].copy()

        # Convert to numeric
        for col in available_indicators:
            df_subset[col] = pd.to_numeric(df_subset[col], errors="coerce").fillna(0.0)

        # Melt
        melted = df_subset.melt(
            id_vars=["organisationunitid", "organisationunitname", "gregorian_date"],
            value_vars=available_indicators,
            var_name="indicator",
            value_name="value",
        )
        all_data.append(melted)

    full_df = pd.concat(all_data, ignore_index=True)
    full_df["year"] = full_df["gregorian_date"].dt.year

    # Map regions
    full_df["region"] = full_df["organisationunitid"].map(id_to_region)

    # Normalize region names for coloring
    full_df["region"] = full_df["region"].apply(
        lambda x: normalize_dhis2_region_name(x) if pd.notna(x) else "Unknown"
    )

    return full_df


def plot_correlation_matrix(df):
    from scipy.cluster.hierarchy import linkage, leaves_list

    out_dir = Path("figures/dhis2_correlations")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Computing correlation matrix...")
    # Pivot to wide format: (facility, year) rows, indicator columns
    pivot_df = (
        df.groupby(["organisationunitid", "year", "indicator"])["value"].sum().unstack()
    )

    pivot_df = pivot_df.rename(columns=INDICATOR_NAMES)

    corr = pivot_df.corr()

    Z = linkage(corr, method="average", metric="euclidean")
    sort_idx = leaves_list(Z)

    corr = corr.iloc[sort_idx, sort_idx]

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")

    mask = np.triu(np.ones_like(corr, dtype=bool))
    plot_corr = corr.iloc[1:, :-1]
    plot_mask = mask[1:, :-1]

    ax = sns.heatmap(
        plot_corr,
        mask=plot_mask,
        cmap="RdYlGn",
        center=0,
        vmax=1,
        vmin=-1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.6, "label": ""},
        annot=False,
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.xlabel("")
    plt.ylabel("")
    plt.title("")

    plt.tight_layout()
    save_path = out_dir / "correlation_matrix.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {save_path}")
    plt.close()


def plot_specific_correlations(df):
    out_dir = Path("figures/dhis2_correlations")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Aggregating by facility and year for scatter plots...")
    # Pivot to wide format: (facility, year, region) rows, indicator columns
    pivot_df = df.pivot_table(
        index=["organisationunitid", "year", "region"],
        columns="indicator",
        values="value",
        aggfunc="sum",
    ).reset_index()

    sns.set_context("talk")
    sns.set_style("white")

    plot_pairs = [
        (
            "MAT_Skilled Birth Attendance",
            "EPI_Children <1 Year Measels- 1st Dose",
            "sba_vs_measles",
        ),
        (
            "MAT_Skilled Birth Attendance",
            "MAT_Births By Caesarean Section",
            "sba_vs_c_section",
        ),
        (
            "EPI_Children <1 Year Measels- 1st Dose",
            "EPI_<1 Year Received 3rd Dose Penta",
            "measles_vs_penta",
        ),
    ]

    print("Plotting scatter plots...")
    for x_col, y_col, fname in plot_pairs:
        if x_col not in pivot_df.columns or y_col not in pivot_df.columns:
            print(f"Skipping {fname}: columns missing")
            continue

        plt.figure(figsize=(8, 8))

        # Prepare data: +1 for log scale
        subset = pivot_df.dropna(subset=[x_col, y_col]).copy()
        subset["adj_x"] = subset[x_col] + 1
        subset["adj_y"] = subset[y_col] + 1

        # Filter unknown regions
        subset = subset[subset["region"].isin(ETHIOPIAN_REGION_COLORS.keys())]

        is_sba_vs_c_section = fname == "sba_vs_c_section"
        show_legend = is_sba_vs_c_section

        label_fs = 14
        tick_fs = 12
        legend_fs = 12
        legend_title_fs = 14

        sns.scatterplot(
            data=subset,
            x="adj_x",
            y="adj_y",
            hue="region",
            palette=ETHIOPIAN_REGION_COLORS,
            alpha=0.5,
            s=20,
            linewidth=0,
            rasterized=True,
            legend=show_legend,
        )

        plt.xscale("log")
        plt.yscale("log")

        # Calculate correlation on raw values
        r = subset[x_col].corr(subset[y_col])

        # Labels
        if x_col == "MAT_Skilled Birth Attendance":
            plt.xlabel("Skilled Birth Attendance", fontsize=label_fs)
        elif x_col == "EPI_Children <1 Year Measels- 1st Dose":
            plt.xlabel("Measles 1st Dose", fontsize=label_fs)
        else:
            plt.xlabel(x_col, fontsize=label_fs)

        if y_col == "EPI_Children <1 Year Measels- 1st Dose":
            plt.ylabel("Measles 1st Dose", fontsize=label_fs)
        elif y_col == "MAT_Births By Caesarean Section":
            plt.ylabel("Caesarean Sections", fontsize=label_fs)
        elif y_col == "EPI_<1 Year Received 3rd Dose Penta":
            plt.ylabel("Penta 3", fontsize=label_fs)
        else:
            plt.ylabel(y_col, fontsize=label_fs)

        plt.xticks(fontsize=tick_fs)
        plt.yticks(fontsize=tick_fs)
        plt.title("")

        # Legend styling
        if show_legend:
            plt.legend(
                loc="upper left",
                borderaxespad=0.0,
                frameon=False,
                title="Region",
                fontsize=legend_fs,
                title_fontsize=legend_title_fs,
            )

        sns.despine()
        plt.tight_layout()

        save_path = out_dir / f"{fname}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path} (r={r:.2f})")
        plt.close()


if __name__ == "__main__":
    df = load_data()
    plot_correlation_matrix(df)
    plot_specific_correlations(df)
