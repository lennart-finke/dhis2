# DHIS2 Coverage Analysis
This repository aims to predict ground-truth coverage rates globally and in Ethiopia from high-quality datasets like DHS, calibrating based on Ethiopia facility-level data from DHIS2.

## Replication
### 1.
- `match_zone_names.py` - Matches DHIS2 zone names to standard geographic boundary data.
- `match_woreda_names.py` - Matches DHIS2 woreda names to geographic boundary data.
- `compute_total_populations.py` - Estimates total populations from WorldPop raster data.
- `compute_population_fractions_un.py` - Computes age-group population fractions using UN statistics.
- `copy_dhis2_data.py` - Ingests and cleans raw DHIS2 CSV datasets by applying basic outlier detection.
- `compute_dhs_estimates.py` - Extracts target indicator survey estimates from raw DHS survey datasets.
- `compute_pma_estimates.py` - Extracts target indicator survey estimates from raw PMA survey datasets.

### 2.
- `compute_dhis2_estimates.py` - Aggregates cleaned DHIS2 facility counts to geographical bounds.
- `find_outliers_ols.py` - Counts statistical outliers in the aggregate DHIS2 counts using OLS.
- `match_facility_names.py` - Identifies and maps individual DHIS2 facility properties against the MFR.
- `compute_woreda_travel_distances.py` - Computes average travel times to nearest health facilities per woreda.

### 3.
- `compute_dhis2_coverage_estimates.py` - Combines aggregate DHIS2 counts with estimated populations to compute raw coverage estimates.

### 4.
- `match_facilities_to_boundaries.py` - Matches DHIS2 facilities to administrative boundaries.
- `compare_survey_dhis2_coverage.py` - Fits beta regression models to calibrate DHIS2 coverage components using survey data predictions.

### 5.
- `compute_calibrated_coverage.py` - Applies the calculated calibration coefficients to generate final, standardized coverage estimates.

### 6.
- `validate_matching.py` - Produces statistical evaluations on name-based vs. GPS-based matching.
- `plot_population_comparison.py` - Generates correlation figures comparing WorldPop populations to census-based populations.
- `validate_woreda_estimates_road_access.py` - Checks correlation between predicted woreda estimates and travel road accessibility times.
- `validate_woreda_estimates_kea2023.py` - Checks estimations against two survey studies.
- `plot_absolute_differences.py` - Visualizes standardized spatial residuals from our regression models.
- `plot_calibrated_coverage_maps.py` - Renders map graphics (choropleths) showing coverage estimates within boundaries.
- `plot_dhis2_correlations.py` - Render statistical summaries analyzing facility count correlations natively spanning various DHIS2 indicators.
- `calculate_morans_i.py` - Analyzes residual patterns to compute spatial autocorrelation metrics (Moran's I) for coverages.