"""
Validate woreda estimates: compare KEA 2023 SBA survey (Sidama) vs DHIS2 calibrated coverage.
"""

import pandas as pd

# KEA 2023 survey: weighted SBA per woreda
kea = pd.read_stata("data/kea2023/Skilled Birth Attendant-Sidama-ETH.dta")
survey = pd.DataFrame(
    [
        {
            "woreda": w,
            "survey_sba": (
                (kea.loc[m, "Del_Skilled_r"] == "Skilled delivery").astype(float)
                * kea.loc[m, "Wr_Wt"]
            ).sum()
            / kea.loc[m, "Wr_Wt"].sum(),
            "n": m.sum(),
        }
        for w in kea["Woreda_Name"].unique()
        for m in [(kea["Woreda_Name"] == w)]
    ]
)
survey["woreda_geo"] = survey["woreda"].map(
    {
        "Wondogenet": "Wondo-Genet",
        "Aleta Wondo": "Aleta Wendo",
        "Aroresa": "Aroresa",
        "Aleta Chuko": "Aleta Chuko",
        "Daela": "Daella",
        "Hawassa Zuriya": "Hawassa Zuria",
    }
)

# Load DHIS2 woreda estimates and aggregate duplicates
dhis2 = pd.read_csv("estimates/woreda_calibrated_coverage_geojson.csv")
sidama_sba = dhis2[
    (dhis2["region"] == "Sidama")
    & (dhis2["indicator"] == "Skilled Birth Attendance")
    & (dhis2["year"] == 2020)
]  # Study was conducted in 2019/20
sidama_sba = sidama_sba.groupby("woreda", as_index=False)[
    ["dhis2_coverage", "calibrated_coverage"]
].mean()

# Compare
merged = survey.merge(
    sidama_sba[["woreda", "dhis2_coverage", "calibrated_coverage"]],
    left_on="woreda_geo",
    right_on="woreda",
    how="left",
)
print("Woreda SBA validation (KEA 2023 vs DHIS2):\n")
print(
    merged[["woreda_x", "n", "survey_sba", "dhis2_coverage", "calibrated_coverage"]]
    .rename(columns={"woreda_x": "woreda"})
    .to_string(index=False, float_format=lambda x: f"{x:.3f}")
)


matched = merged.dropna(subset=["calibrated_coverage"])
print(
    f"\nMatched: {len(matched)}/{len(survey)}, Pearson Corr: {matched['survey_sba'].corr(matched['calibrated_coverage'], method='pearson'):.3f}, Spearman Corr: {matched['survey_sba'].corr(matched['calibrated_coverage'], method='spearman'):.3f}, MAE: {(matched['survey_sba'] - matched['calibrated_coverage']).abs().mean():.3f}"
)


# Ayele et al. 2019 validation
print("\n" + "=" * 80 + "\n")
print("Validation: Ayele et al. 2019 (Gura Dhamole Woreda, 2017)")
print("Source: https://link.springer.com/article/10.1186/s12889-019-7818-6")

ayele_survey = pd.DataFrame(
    [
        {
            "woreda": "Gura Damole",  # Spelled Gura Damole in our CSV
            "year": 2017,
            "survey_sba": 0.292,
            "n": 402,
        }
    ]
)

# Filter DHIS2 estimates for the matched woreda/year
# Note: Data starts from 2018, so we compare 2017 survey vs 2018 estimates
ayele_dhis2 = dhis2[
    (dhis2["woreda"] == "Gura Damole")
    & (dhis2["indicator"] == "Skilled Birth Attendance")
    & (dhis2["year"] == 2018)
]

merged_ayele = ayele_survey.merge(
    ayele_dhis2[["woreda", "dhis2_coverage", "calibrated_coverage"]],
    on="woreda",
    how="left",
)
merged_ayele = merged_ayele.rename(
    columns={
        "dhis2_coverage": "dhis2_2018",
        "calibrated_coverage": "calibrated_2018",
    }
)

print(
    merged_ayele[
        ["woreda", "n", "survey_sba", "dhis2_2018", "calibrated_2018"]
    ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
)
