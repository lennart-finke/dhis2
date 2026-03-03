"""
Shared statistics and plotting utilities for comparing estimates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
from statsmodels.othermod.betareg import BetaModel

from dhis2_utils import get_region_color

from beta_ecological import EcologicalBetaModel


def regression_ecological(
    data,
    x_col,
    y_col,
    weight_col,
    woreda_data,
    woreda_x_col,
    woreda_region_col,
    woreda_year_col,
    region_col=None,
    dataset_col=None,
    woreda_weight_col=None,
    print_summary=True,
):
    """
    Compute ecological-inference-corrected beta regression using EcologicalBetaModel.

    Fits parameters by applying the link function at woreda level, then
    averaging to regional predictions for comparison with regional observations.
    This corrects for bias when parameters fitted on regional data are later
    applied at sub-regional (woreda) level.

    Uses the EcologicalBetaModel from beta_ecological.py which provides:
    - Analytical score and Hessian for faster, more robust optimization
    - Proper statsmodels integration with standard inference (SE, CI, p-values)
    - Support for population-weighted aggregation at woreda level

    Args:
        data: DataFrame containing regional-level observations
        x_col: Column name for regional X (used for reference, actual fitting uses woreda data)
        y_col: Column name for Y (regional survey coverage)
        weight_col: Column name for regional weights
        woreda_data: DataFrame containing woreda-level DHIS2 data
        woreda_x_col: Column name for woreda-level X (DHIS2 coverage)
        woreda_region_col: Column name for region identifier in woreda data
        woreda_year_col: Column name for year in woreda data
        region_col: Optional column for region (categorical, fixed effects)
        dataset_col: Optional column for dataset (categorical, fixed effects)
        woreda_weight_col: Optional column for woreda-level population weights.
                          If provided, regional means are population-weighted averages.

    Returns:
        dict: {'r2': r2, 'm': slope, 'b': intercept, 'method': 'ecological_beta',
               'results': EcologicalBetaResults}
              If region_col: also includes 'region_coefs': {region: coef}
              If dataset_col: also includes 'dataset_coefs': {dataset: coef}
    """
    # Prepare regional data
    cols = [x_col, y_col, weight_col, "region", "year"]
    if region_col:
        cols.append(region_col)
    if dataset_col:
        cols.append(dataset_col)
    cols = list(dict.fromkeys(cols))  # Remove duplicates

    df = data[cols].dropna().copy()

    assert len(df) >= 2, "Not enough data for regression"

    # Build woreda-level design matrix and mapping
    # We need to construct:
    # - Y_reg: regional observations (n_regions,)
    # - X_long: woreda-level design matrix (n_woredas, k)
    # - group_idx: mapping woredas to regions (n_woredas,)
    # - woreda_weights: optional population weights (n_woredas,)

    Y_reg = []
    X_long_parts = []
    group_idx_parts = []
    woreda_weights_parts = []
    region_obs_idx = 0

    # Track unique datasets for fixed effects
    unique_datasets = sorted(df[dataset_col].unique()) if dataset_col else []

    # Track unique regions for fixed effects
    unique_regions = sorted(df[region_col].unique()) if region_col else []
    region_to_dummy_idx = (
        {r: i for i, r in enumerate(unique_regions[1:])} if region_col else {}
    )
    n_region_dummies = len(unique_regions) - 1 if region_col else 0

    # Reference category is the first one (will have coefficient 0)
    dataset_to_dummy_idx = (
        {d: i for i, d in enumerate(unique_datasets[1:])} if dataset_col else {}
    )

    n_dataset_dummies = len(unique_datasets) - 1 if dataset_col else 0

    # Build column names for the design matrix
    col_names = ["const", "dhis2_coverage"]

    # Add region dummies
    for r in unique_regions[1:]:
        col_names.append(f"region_{r}")

    for d in unique_datasets[1:]:
        col_names.append(f"dataset_{d}")

    for _, row in df.iterrows():
        region = row["region"]
        year = row["year"]

        # Get woredas for this region-year
        woreda_mask = (woreda_data[woreda_region_col] == region) & (
            woreda_data[woreda_year_col] == year
        )
        woreda_subset = woreda_data[woreda_mask]

        assert (
            len(woreda_subset) > 0
        ), f"No woreda data for region {region} and year {year}"

        # Get woreda X values
        x_woredas = woreda_subset[woreda_x_col].dropna()
        if len(x_woredas) == 0:
            continue

        n_woredas = len(x_woredas)

        # Build design matrix for this region's woredas
        # Columns: [intercept, x1, (x2), (dataset dummies)]
        n_cols = 2  # intercept + x1
        n_cols += n_region_dummies
        n_cols += n_dataset_dummies

        X_woreda = np.zeros((n_woredas, n_cols))
        X_woreda[:, 0] = 1.0  # intercept
        X_woreda[:, 1] = x_woredas.values  # x1 (DHIS2 coverage)

        col_idx = 2

        # Region fixed effects (dummy encoding)
        if region_col:
            region_val = row[region_col]
            if region_val in region_to_dummy_idx:
                dummy_idx = col_idx + region_to_dummy_idx[region_val]
                X_woreda[:, dummy_idx] = 1.0

        # Advance col_idx past region dummies
        col_idx += n_region_dummies

        # Dataset fixed effects (dummy encoding)
        if dataset_col:
            dataset_val = row[dataset_col]
            if dataset_val in dataset_to_dummy_idx:
                dummy_idx = col_idx + dataset_to_dummy_idx[dataset_val]
                X_woreda[:, dummy_idx] = 1.0

        # Get woreda weights if provided
        if woreda_weight_col and woreda_weight_col in woreda_subset.columns:
            w_woredas = woreda_subset.loc[x_woredas.index, woreda_weight_col].values
        else:
            w_woredas = np.ones(n_woredas)

        # Store
        Y_reg.append(row[y_col])
        X_long_parts.append(X_woreda)
        group_idx_parts.append(np.full(n_woredas, region_obs_idx, dtype=int))
        woreda_weights_parts.append(w_woredas)
        region_obs_idx += 1

    if region_obs_idx < 2:
        raise ValueError(
            "Not enough regions with woreda data for ecological regression"
        )

    # Assemble arrays
    Y_reg = np.array(Y_reg)
    X_long = np.vstack(X_long_parts)
    group_idx = np.concatenate(group_idx_parts)
    woreda_weights = np.concatenate(woreda_weights_parts) if woreda_weight_col else None

    # Convert to DataFrame with proper column names for labeled output
    X_long_df = pd.DataFrame(X_long, columns=col_names)

    # Fit the EcologicalBetaModel
    model = EcologicalBetaModel(
        endog=Y_reg,
        exog=X_long_df,
        group_idx=group_idx,
        weights=woreda_weights,
    )

    results = model.fit(disp=False)
    if print_summary:
        print(results.summary())

    # Extract parameters
    params = results.params
    intercept = params[0]
    slope = params[1]

    result = {
        "r2": results.prsquared,
        "m": slope,
        "b": intercept,
        "method": "ecological_beta",
        "results": results,  # Full results object for advanced use
        "region_coefs": {},
    }

    # Extract x2 coefficient if present
    col_idx = 2

    # Extract region fixed effects
    if region_col:
        region_coefs = {}
        for r in unique_regions:
            if r in region_to_dummy_idx:
                region_coefs[r] = params[col_idx + region_to_dummy_idx[r]]
            else:
                region_coefs[r] = 0.0  # Reference region
        result["region_coefs"] = region_coefs
        col_idx += n_region_dummies
    else:
        result["region_coefs"] = {}

    # Extract dataset fixed effects
    if dataset_col:
        dataset_coefs = {}
        for d in unique_datasets:
            if d in dataset_to_dummy_idx:
                dataset_coefs[d] = params[col_idx + dataset_to_dummy_idx[d]]
            else:
                dataset_coefs[d] = 0.0  # Reference dataset
        result["dataset_coefs"] = dataset_coefs

    return result


class WeightedBetaModel(BetaModel):
    """
    Beta regression model with sample weights support.

    The standard statsmodels BetaModel does not support sample weighting.
    This subclass incorporates weights into the log-likelihood, score, and
    hessian calculations to enable weighted maximum likelihood estimation.

    Weights are interpreted as inverse-variance weights (analytic weights),
    where higher weights indicate more precise/reliable observations.
    """

    def __init__(
        self,
        endog,
        exog,
        weights=None,
        exog_precision=None,
        link=None,
        link_precision=None,
        **kwds,
    ):
        """
        Initialize the weighted beta model.

        Parameters
        ----------
        endog : array_like
            1-d array of endogenous response variable.
        exog : array_like
            A nobs x k array where nobs is the number of observations and k
            is the number of regressors.
        weights : array_like, optional
            1-d array of sample weights. Should be positive. If None, equal
            weights are assumed.
        exog_precision : array_like, optional
            Regressors for the precision parameter.
        link : Link, optional
            Link function for the mean model. Defaults to Logit.
        link_precision : Link, optional
            Link function for the precision model. Defaults to Log.
        """
        # Build parent kwargs, only including non-None link arguments
        parent_kwds = {"exog_precision": exog_precision, **kwds}
        if link is not None:
            parent_kwds["link"] = link
        if link_precision is not None:
            parent_kwds["link_precision"] = link_precision

        super().__init__(endog, exog, **parent_kwds)

        if weights is None:
            self.weights = np.ones(len(endog))
        else:
            self.weights = np.asarray(weights)
            # Normalize weights to sum to sample size for numerical stability
            self.weights = self.weights * len(endog) / self.weights.sum()

    def loglike(self, params):
        """
        Weighted log-likelihood of model at params.

        Returns the weighted sum of observation-level log-likelihoods.
        """
        return (self.weights * self.loglikeobs(params)).sum()

    def score(self, params):
        """
        Weighted score vector of the log-likelihood.

        Returns the first derivative of weighted loglikelihood function.
        """
        sf1, sf2 = self.score_factor(params)

        # Apply weights to score factors
        sf1_weighted = sf1 * self.weights
        sf2_weighted = sf2 * self.weights

        d1 = np.dot(sf1_weighted, self.exog)
        d2 = np.dot(sf2_weighted, self.exog_precision)
        return np.concatenate((d1, d2))

    def hessian(self, params, observed=None):
        """
        Weighted Hessian, second derivative of loglikelihood function.

        Returns the weighted observed or expected information matrix.
        """
        if self.hess_type == "eim":
            observed = False
        else:
            observed = True
        _, hf = self.score_hessian_factor(
            params, return_hessian=True, observed=observed
        )

        hf11, hf12, hf22 = hf

        # Apply weights to hessian factors
        hf11_weighted = hf11 * self.weights
        hf12_weighted = hf12 * self.weights
        hf22_weighted = hf22 * self.weights

        # Elementwise product for each row (observation)
        d11 = (self.exog.T * hf11_weighted).dot(self.exog)
        d12 = (self.exog.T * hf12_weighted).dot(self.exog_precision)
        d22 = (self.exog_precision.T * hf22_weighted).dot(self.exog_precision)
        return np.block([[d11, d12], [d12.T, d22]])


def regression(
    data,
    x_col,
    y_col,
    weight_col,
    method="linear",
    region_col=None,
    dataset_col=None,
    print_summary=True,
):
    """
    Compute regression stats using statsmodels.

    Args:
        data: DataFrame containing the data
        x_col: Column name for X1 (primary predictor, e.g. DHIS2 coverage)
        y_col: Column name for Y (response/dependent variable)
        weight_col: Column name for weights (inverse of variance)
        method: "linear" (WLS) or "beta" (Beta Regression)
        region_col: Optional column for region (categorical, dummy encoded)
        dataset_col: Optional column for dataset (categorical, dummy encoded)

    Returns:
        dict: {'r2': r2, 'm': slope, 'b': intercept, 'method': method}
              If region_col: also includes 'region_coefs': {region: coef}
              If dataset_col: also includes 'dataset_coefs': {dataset: coef}
              For beta regression, r2 is pseudo-R2, coefficients are logit-link.
    """
    cols = [x_col, y_col, weight_col]
    if region_col:
        cols.append(region_col)
    if dataset_col:
        cols.append(dataset_col)

    df = data[cols].dropna().copy()

    assert len(df) >= 2, "Not enough data for regression"

    weights = df[weight_col]
    x = df[x_col]
    y = df[y_col]

    # Build design matrix
    design_parts = [
        pd.Series(1.0, index=df.index, name="const"),
        x.rename("x1").astype(float),
    ]

    if region_col:
        # Create dummy variables for region (drop_first=True for identifiability)
        region_dummies = pd.get_dummies(
            df[region_col], prefix="region", drop_first=True, dtype=float
        )
        design_parts.append(region_dummies)

    if dataset_col:
        # Create dummy variables for dataset (drop_first=True for identifiability)
        dataset_dummies = pd.get_dummies(
            df[dataset_col], prefix="dataset", drop_first=True, dtype=float
        )
        design_parts.append(dataset_dummies)

    if len(design_parts) > 2:  # Has additional variables
        X = pd.concat(design_parts, axis=1)
    else:
        X = sm.add_constant(x)

    if method == "linear":
        model = sm.WLS(y, X, weights=weights)
        results = model.fit()
        r2 = results.rsquared
        b = results.params["const"]
        m = (
            results.params["x1"]
            if (region_col or dataset_col)
            else results.params[x_col]
        )

    elif method == "beta":
        y_clipped = y.clip(1e-2, 1 - 1e-2)
        model = WeightedBetaModel(y_clipped, X, weights=weights)
        results = model.fit(disp=0)
        r2 = results.prsquared
        b = results.params["const"]
        m = (
            results.params["x1"]
            if (region_col or dataset_col)
            else results.params[x_col]
        )
    else:
        raise ValueError(f"Unknown regression method: {method}")
    if print_summary:
        print(results.summary())
    result = {"r2": r2, "m": m, "b": b, "method": method, "results": results}

    if region_col:
        # Extract region coefficients
        region_coefs = {}
        regions = df[region_col].unique()
        for col in results.params.index:
            if col.startswith("region_"):
                region_name = col.replace("region_", "")
                region_coefs[region_name] = results.params[col]
        # Reference region (dropped) has coefficient 0
        for r in regions:
            if r not in region_coefs:
                region_coefs[r] = 0.0
        result["region_coefs"] = region_coefs

    if dataset_col:
        # Extract dataset coefficients
        dataset_coefs = {}
        datasets = df[dataset_col].unique()
        for col in results.params.index:
            if col.startswith("dataset_"):
                dataset_name = col.replace("dataset_", "")
                dataset_coefs[dataset_name] = results.params[col]
        # Reference dataset (dropped) has coefficient 0
        for d in datasets:
            if d not in dataset_coefs:
                dataset_coefs[d] = 0.0
        result["dataset_coefs"] = dataset_coefs

    return result


def predict_coverage(
    value,
    slope,
    intercept,
    method="linear",
    region=None,
    region_coefs=None,
    dataset=None,
    dataset_coefs=None,
):
    """
    Predict coverage based on value(s), slope(s), intercept and method.

    Args:
        value: Primary predictor value (e.g. DHIS2 coverage)
        slope: Slope for primary predictor
        intercept: Intercept
        method: "linear" or "beta"
        region: Optional region name for region-specific adjustment
        region_coefs: Dict mapping region names to their coefficients
        dataset: Optional dataset name for dataset-specific adjustment
        dataset_coefs: Dict mapping dataset names to their coefficients
        dataset_coefs: Dict mapping dataset names to their coefficients
    """
    linear_pred = value * slope + intercept
    if region is not None and region_coefs is not None:
        region_coef = region_coefs.get(region, 0.0)
        linear_pred = linear_pred + region_coef
    if dataset is not None and dataset_coefs is not None:
        dataset_coef = dataset_coefs.get(dataset, 0.0)
        linear_pred = linear_pred + dataset_coef

    if method == "linear":
        return linear_pred
    elif method == "beta" or method == "ecological_beta":
        from scipy.special import expit

        return expit(linear_pred)
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_scatter_comparison(
    data,
    x_col,
    y_col,
    label_col,
    year_col,
    title,
    x_label,
    y_label,
    output_path,
    weight_col=None,
    size_scale_factor=100,
    min_size=30,
    max_size=300,
    marker_col=None,
    marker_map=None,
    regression_line_params=None,
    color_col=None,
):
    """
    Create and save a scatter plot comparison.

    Args:
        data: DataFrame containing the data
        x_col: X column name
        y_col: Y column name
        label_col: Column to use for legend labels (e.g., Region or Zone)
        year_col: Column to use for point annotations (e.g., Year)
        title: Plot title
        x_label: X axis label
        y_label: Y axis label
        output_path: Path to save the figure
        weight_col: Optional column for sizing points
        size_scale_factor: Base size if no weights, or scale factor for weights
        min_size: Minimum point size when weighting
        max_size: Maximum point size when weighting
        marker_col: Optional column for marker styles
        marker_map: Dictionary mapping marker_col values to matplotlib marker characters
        regression_line_params: Dict with 'slope', 'intercept', 'method' to plot fit line
        color_col: Optional column for determining colors (defaults to label_col).
                   Useful for coloring zones by their region.
    """
    plt.figure(figsize=(8, 8))

    # Determine color column (default to label_col)
    if color_col is None:
        color_col = label_col

    # Unique labels for grouping
    sorted(data[label_col].unique())

    # Build color mapping based on color_col values
    unique_color_keys = data[color_col].unique()
    colors = {}
    for key in unique_color_keys:
        colors[key] = get_region_color(key)

    # Prepare sizes
    if weight_col:
        weights = data[weight_col].fillna(0)
        # Use square root scaling for better perceptual linearity if weights vary
        if weights.max() > weights.min():
            sqrt_w = np.sqrt(weights)
            min_sqrt = sqrt_w.min()
            max_sqrt = sqrt_w.max()
            if max_sqrt > min_sqrt:
                sizes = (sqrt_w - min_sqrt) / (max_sqrt - min_sqrt) * (
                    max_size - min_size
                ) + min_size
            else:
                sizes = pd.Series([size_scale_factor] * len(data), index=data.index)
        else:
            sizes = pd.Series([size_scale_factor] * len(data), index=data.index)
    else:
        sizes = pd.Series([50] * len(data), index=data.index)  # Default fixed size

    # Track added labels to avoid duplicates in legend
    added_labels = set()

    # Plot each group - group by color_col to batch by region color
    unique_color_groups = sorted(data[color_col].unique())

    for color_key in unique_color_groups:
        mask_color = data[color_col] == color_key
        subset_color = data[mask_color]
        point_color = colors.get(color_key, "#808080")

        if marker_col and marker_map:
            # Plot subset for each marker type
            for m_key, m_style in marker_map.items():
                mask_marker = subset_color[marker_col] == m_key
                sub = subset_color[mask_marker]
                if sub.empty:
                    continue

                sub_sizes = sizes[mask_color][mask_marker]

                # Use color_key (region) as legend label
                if color_key not in added_labels:
                    lbl = color_key
                    added_labels.add(color_key)
                else:
                    lbl = None

                plt.scatter(
                    sub[x_col],
                    sub[y_col],
                    c=point_color,
                    label=lbl,
                    marker=m_style,
                    s=sub_sizes,
                    alpha=0.7,
                )
        else:
            subset_sizes = sizes[mask_color]

            if color_key not in added_labels:
                lbl = color_key
                added_labels.add(color_key)
            else:
                lbl = None

            plt.scatter(
                subset_color[x_col],
                subset_color[y_col],
                c=point_color,
                label=lbl,
                s=subset_sizes,
                alpha=0.7,
            )

    # Add marker legend entries
    if marker_map:
        for m_key, m_style in marker_map.items():
            plt.scatter([], [], c="k", alpha=0.7, marker=m_style, label=m_key)

    # Plot regression line if params provided
    if regression_line_params:
        slope = regression_line_params.get("slope")
        intercept = regression_line_params.get("intercept")
        method = regression_line_params.get("method", "linear")

        if slope is not None and intercept is not None:
            # Create smooth line (base model without region adjustment)
            x_vals = np.linspace(data[x_col].min(), data[x_col].max(), 100)
            y_vals = predict_coverage(x_vals, slope, intercept, method)

            plt.plot(
                x_vals, y_vals, "k-", alpha=0.5, linewidth=2, label="Fit", zorder=5
            )

    # Annotations
    if year_col:
        for idx, row in data.iterrows():
            if pd.notna(row[x_col]) and pd.notna(row[y_col]):
                txt = f"{int(row[year_col])}" if pd.notna(row[year_col]) else ""
                plt.annotate(
                    txt,
                    (row[x_col], row[y_col]),
                    fontsize=8,
                    xytext=(3, 3),
                    textcoords="offset points",
                )

    # Reference line (identity line) if assuming y ~= x, or just fitting range
    # Creating a 1-1 line is standard for "comparison" plots
    all_vals = pd.concat([data[x_col], data[y_col]])
    if not all_vals.empty:
        min_val = all_vals.min()
        max_val = all_vals.max()
        plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, alpha=0.5)

    plt.xlabel(x_label, fontsize=9)
    plt.ylabel(y_label, fontsize=9)
    plt.title(title, fontsize=10)
    plt.legend(fontsize=8, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    # Also save in pdf, in pdf subfolder
    pdf_path = output_path.parent / "pdf" / output_path.name.replace(".png", ".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close()
