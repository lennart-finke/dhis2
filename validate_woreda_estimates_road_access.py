"""
Validate woreda estimates: correlate SBA coverage with facility travel distances.
Compares both population-weighted and unweighted travel distance metrics
for motorized transport mode.
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Load travel distances
travel = pd.read_csv("estimates/woreda_travel_distances.csv")

# Load woreda SBA coverage (latest year)
coverage = pd.read_csv("estimates/woreda_calibrated_coverage_geojson_latest.csv")
sba = coverage[coverage["indicator"] == "Skilled Birth Attendance"].copy()
sba = sba.groupby("woreda", as_index=False)["calibrated_coverage"].mean()

# Merge on woreda name
merged = travel.merge(sba, on="woreda", how="inner")


# Filter for woredas with motorized data
merged_motorized = merged[merged["motorized_weighted_mean_hrs"].notna()].copy()
print(f"Matched {len(merged_motorized)} woredas with motorized travel and SBA data")
print("=" * 70)


# Motorized: Population-Weighted
print("\nMotorized - Population-Weighted:")
r_motor_w, p_motor_w = stats.pearsonr(
    merged_motorized["motorized_weighted_mean_hrs"],
    merged_motorized["calibrated_coverage"],
)
print(f"  Pearson r:  {r_motor_w:.3f}, p-value: {p_motor_w:.2e}")
rho_motor_w, p_rho_motor_w = stats.spearmanr(
    merged_motorized["motorized_weighted_mean_hrs"],
    merged_motorized["calibrated_coverage"],
)
print(f"  Spearman ρ: {rho_motor_w:.3f}, p-value: {p_rho_motor_w:.2e}")

# Motorized: Unweighted
print("\nMotorized - Unweighted:")
r_motor_u, p_motor_u = stats.pearsonr(
    merged_motorized["motorized_unweighted_mean_hrs"],
    merged_motorized["calibrated_coverage"],
)
print(f"  Pearson r:  {r_motor_u:.3f}, p-value: {p_motor_u:.2e}")
rho_motor_u, p_rho_motor_u = stats.spearmanr(
    merged_motorized["motorized_unweighted_mean_hrs"],
    merged_motorized["calibrated_coverage"],
)
print(f"  Spearman ρ: {rho_motor_u:.3f}, p-value: {p_rho_motor_u:.2e}")

print("\n" + "=" * 70)
print("Summary:")

print("\nMotorized mode:")
print(
    f"  Weighted has {'stronger' if abs(r_motor_w) > abs(r_motor_u) else 'weaker'} Pearson correlation (Δr = {abs(r_motor_w) - abs(r_motor_u):.3f})"
)
print(
    f"  Weighted has {'stronger' if abs(rho_motor_w) > abs(rho_motor_u) else 'weaker'} Spearman correlation (Δρ = {abs(rho_motor_w) - abs(rho_motor_u):.3f})"
)

# Determine best predictor
correlations = {
    "Motorized Weighted": abs(rho_motor_w),
    "Motorized Unweighted": abs(rho_motor_u),
}
best_predictor = max(correlations, key=correlations.get)
print(
    f"\nStrongest predictor (by Spearman ρ): {best_predictor} (ρ = {correlations[best_predictor]:.3f})"
)

# Plot only Weighted Motorized
fig, ax = plt.subplots(figsize=(8, 6))

# Plot: Motorized - Population-weighted
ax.scatter(
    merged_motorized["motorized_weighted_mean_hrs"],
    merged_motorized["calibrated_coverage"],
    alpha=0.8,
    s=15,
    c="black",
)
ax.set_xlabel("Population-weighted mean travel time (hours)", fontsize=12)
ax.set_ylabel("Calibrated SBA coverage", fontsize=12)

# Force y axis from 0 to 1
ax.set_ylim(0, 1.005)

# Grid
ax.grid(True, alpha=0.3, which="both")

fig.tight_layout()
fig.savefig("figures/sba_vs_travel_distance.png", dpi=150, bbox_inches="tight")
print("\nSaved figures/sba_vs_travel_distance.png")
