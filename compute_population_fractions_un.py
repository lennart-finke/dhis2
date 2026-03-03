"""
Compute population fractions from UN population and fertility data.
Saves to estimates/population_fractions_un.csv.
"""

from pathlib import Path
import pandas as pd
import numpy as np

pop_df = pd.read_csv("data/population/un_pop.csv")
fert_df = pd.read_csv("data/population/un_fertility.csv")

# Filter for Ethiopia and medium variant
pop_df = pop_df[
    (pop_df["Location"] == "Ethiopia") & (pop_df["Variant"] == "Medium")
].copy()
fert_df = fert_df[
    (fert_df["Location"] == "Ethiopia") & (fert_df["Variant"] == "Medium")
].copy()

# Calculate fractions by year
results = []

for year in sorted(pop_df["Time"].unique()):
    pop_year = pop_df[pop_df["Time"] == year].copy()
    fert_year = fert_df[fert_df["Time"] == year].copy()

    # Population is in thousands
    total_pop = pop_year["PopTotal"].sum()

    # <1 Year: age 0
    pop_under_1 = pop_year[pop_year["AgeGrpStart"] == 0]["PopTotal"].sum()
    frac_under_1 = pop_under_1 / total_pop

    # 6-59 Months: ages 0.5 to 4 years
    # This includes half of age 0, plus ages 1-4
    pop_6_59_months = (
        pop_year[pop_year["AgeGrpStart"] == 0]["PopTotal"].sum() / 2
        + pop_year[(pop_year["AgeGrpStart"] >= 1) & (pop_year["AgeGrpStart"] <= 4)][
            "PopTotal"
        ].sum()
    )
    frac_6_59_months = pop_6_59_months / total_pop

    # 24-59 Months: ages 2-4 years
    pop_24_59_months = pop_year[
        (pop_year["AgeGrpStart"] >= 2) & (pop_year["AgeGrpStart"] <= 4)
    ]["PopTotal"].sum()
    frac_24_59_months = pop_24_59_months / total_pop

    # Births
    # Total births in the year
    total_births = fert_year["Births"].sum()

    avg_births = total_births
    frac_births = avg_births / total_pop

    results.append(
        {
            "year": year,
            "age_group": "<1 Year",
            "fraction": frac_under_1,
            "source": "UN WPP",
        }
    )
    results.append(
        {
            "year": year,
            "age_group": "6-59 Months",
            "fraction": frac_6_59_months,
            "source": "UN WPP",
        }
    )
    results.append(
        {
            "year": year,
            "age_group": "24-59 Months",
            "fraction": frac_24_59_months,
            "source": "UN WPP",
        }
    )
    results.append(
        {
            "year": year,
            "age_group": "Births",
            "fraction": frac_births,
            "source": "UN WPP",
        }
    )

df = pd.DataFrame(results)

# Extend to 2026 using linear regression
extensions = []
for group, data in df.groupby("age_group"):
    slope, intercept = np.polyfit(data["year"], data["fraction"], 1)
    max_year = data["year"].max()
    for y in range(max_year + 1, 2027):
        extensions.append(
            {
                "year": y,
                "age_group": group,
                "fraction": slope * y + intercept,
                "source": "Linear Regression",
            }
        )

df = pd.concat([df, pd.DataFrame(extensions)], ignore_index=True)

# Save
out_path = Path("estimates/population_fractions_un.csv")
df.to_csv(out_path, index=False)

print(f"Wrote {len(df)} rows to {out_path}")
print(f"Years: {sorted(df['year'].unique())}")
