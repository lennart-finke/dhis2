"""
Compare DHS-based vs UN-based population fractions.
"""

import pandas as pd

dhs_df = pd.read_csv("estimates/population_fractions_dhs.csv")
un_df = pd.read_csv("estimates/population_fractions_un.csv")

# Show DHS fractions (2016 baseline)
print("DHS-based fractions (EDHS 2016 static):")
print(dhs_df.to_string(index=False))

# Show UN fractions for 2016 for comparison
print("\n\nUN-based fractions (2016 for comparison):")
un_2016 = un_df[un_df["year"] == 2016].copy()
print(un_2016.to_string(index=False))

# Show UN trend over time
print("\n\nUN-based fractions over time:")
pivot = un_df.pivot(index="year", columns="age_group", values="fraction")
print(pivot.to_string())

# Calculate differences
print("\n\nDifference (UN 2016 - DHS 2016):")
comparison = []
for _, row in dhs_df.iterrows():
    age_group = row["age_group"]
    dhs_frac = row["fraction"]
    un_frac = un_2016[un_2016["age_group"] == age_group]["fraction"].values[0]
    diff = un_frac - dhs_frac
    pct_diff = (diff / dhs_frac) * 100
    comparison.append(
        {
            "age_group": age_group,
            "DHS": dhs_frac,
            "UN": un_frac,
            "Difference": diff,
            "% Difference": pct_diff,
        }
    )

comp_df = pd.DataFrame(comparison)
print(comp_df.to_string(index=False))
