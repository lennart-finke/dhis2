"""
Ecological Beta Regression for modeling rates and proportions with sub-regional aggregation.

This model addresses the ecological fallacy that arises when fitting regression on region
level but inferring on sub-regional (woreda/zone) level. Instead of applying the inverse
link function to aggregated covariates:

    μ_i = g^{-1}(Σ_k β_k Σ_w x_{iwk})

we aggregate after the nonlinear transformation:

    μ_i = Σ_w g^{-1}(Σ_k x_{iwk} β_k)

This avoids the systematic variance underestimation caused by Jensen's inequality.

References
----------
Ferrari, S.L.P. and Cribari-Neto, F. (2004). "Beta regression for modelling rates
and proportions." Journal of Applied Statistics, 31(7), 799-815.
"""

import numpy as np
from scipy.special import gammaln as lgamma, psi as digamma, polygamma
from statsmodels.othermod.betareg import BetaModel


from statsmodels.base.model import (
    GenericLikelihoodModel,
    GenericLikelihoodModelResults,
    _LLRMixin,
)
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly

LOGIT_LINK = families.links.Logit()
LOG_LINK = families.links.Log()


def trigamma(x):
    """First derivative of the digamma function (second derivative of log-gamma)."""
    return polygamma(1, x)


class EcologicalBetaModel(GenericLikelihoodModel):
    """Ecological Beta Regression with sub-regional aggregation.

    This model is designed for situations where the response variable is observed at
    an aggregate level (e.g., region) but covariates are available at a finer level
    (e.g., woreda/zone). The model aggregates individual predictions after applying
    the inverse link function, avoiding the ecological fallacy.

    Parameters
    ----------
    endog : array_like
        1d array of endogenous response variable (region-level observations).
        Values must be in (0, 1).
    exog : array_like
        A (n_subunits x k) array of covariates at the sub-regional level.
    group_idx : array_like
        1d integer array of length n_subunits mapping each sub-unit to its
        parent region. Values should be 0-indexed region indices.
    weights : array_like, optional
        1d array of weights for each subunit. Weights are normalized within
        each region to sum to 1. If None, equal weights are used (simple average).
        This allows population-weighted aggregation.
    exog_precision : array_like, optional
        2d array of variables for the precision (region-level).
        Default is intercept only.
    link : link
        Any link in sm.families.links for mean, should have range in
        interval [0, 1]. Default is logit-link.
    link_precision : link
        Any link in sm.families.links for precision, should have
        range in positive line. Default is log-link.
    **kwds : extra keywords
        Keyword options passed to parent class.

    Notes
    -----
    The model is parameterized as:

        μ_i = Σ_{w ∈ region_i} weights_{iw} · g^{-1}(x_{iw}' β)

    where g is the link function (default logit), weights are normalized to sum to 1
    within each region, and the response y_i ~ Beta(μ_i φ, (1-μ_i) φ).

    The log-likelihood, score, and Hessian are derived following Ferrari & Cribari-Neto (2004)
    but modified for the ecological aggregation structure.
    """

    def __init__(
        self,
        endog,
        exog,
        group_idx,
        weights=None,
        exog_precision=None,
        link=LOGIT_LINK,
        link_precision=LOG_LINK,
        **kwds,
    ):
        # Capture variable names if available in exog (e.g. pandas DataFrame)
        exog_names = None
        if hasattr(exog, "columns"):
            exog_names = list(exog.columns)

        endog = np.asarray(endog)
        exog = np.asarray(exog)
        group_idx = np.asarray(group_idx, dtype=int)

        # Validate endog is in (0, 1)
        assert np.all((0 < endog) & (endog < 1)), "endog must be in (0, 1)"

        self.n_regions = len(endog)
        self.n_subunits = len(exog)

        # Validate group_idx
        assert (
            len(group_idx) == self.n_subunits
        ), "group_idx must have same length as exog"
        assert group_idx.min() >= 0, "group_idx must be non-negative"
        assert group_idx.max() < self.n_regions, "group_idx values must be < n_regions"

        # Store group mapping
        self.group_idx = group_idx

        # Weights
        if weights is None:
            weights = np.ones(self.n_subunits)
        else:
            weights = np.asarray(weights)
            assert (
                len(weights) == self.n_subunits
            ), "weights must have same length as exog"
            assert np.all(weights >= 0), "weights must be non-negative"

        # Normalize weights within each region
        self.weights = np.zeros(self.n_subunits)
        for i in range(self.n_regions):
            mask = group_idx == i
            region_sum = weights[mask].sum()
            if region_sum > 0:
                self.weights[mask] = weights[mask] / region_sum
            else:
                # If all weights are zero, use equal weights
                n_in_region = mask.sum()
                self.weights[mask] = 1.0 / n_in_region

        # Build aggregation structure
        self._build_aggregation_matrix()

        # Precision exog (region-level)
        if exog_precision is None:
            extra_names = ["precision"]
            exog_precision = np.ones((self.n_regions, 1), dtype=float)
        else:
            exog_precision = np.asarray(exog_precision)
            extra_names = [
                "precision-%s" % zc
                for zc in (
                    exog_precision.columns
                    if hasattr(exog_precision, "columns")
                    else range(1, exog_precision.shape[1] + 1)
                )
            ]

        kwds["extra_params_names"] = extra_names

        # statsmodels expects exog to have same length as endog
        # We pass a placeholder exog to parent and store the real subunit-level exog separately
        self.exog_long = exog  # subunit-level covariates (n_subunits x k)

        # Create region-level placeholder exog for statsmodels compatibility
        # We'll use mean of subunit covariates as a reasonable placeholder
        exog_region = np.zeros((self.n_regions, exog.shape[1]))
        for i in range(self.n_regions):
            indices = np.where(group_idx == i)[0]
            if len(indices) > 0:
                exog_region[i] = exog[indices].mean(axis=0)

        super().__init__(endog, exog_region, exog_precision=exog_precision, **kwds)

        # Restore variable names if they were available
        if exog_names is not None:
            # Update the mean parameter names (the first k_mean names)
            n_mean = len(exog_names)
            # self.exog_names contains both mean and precision param names
            if len(self.exog_names) >= n_mean:
                self.exog_names[:n_mean] = exog_names
                # Also update data module names to ensure consistency
                if hasattr(self, "data") and hasattr(self.data, "xnames"):
                    self.data.xnames[:n_mean] = exog_names

        self.link = link
        self.link_precision = link_precision
        self.nobs = self.n_regions
        self.k_mean = self.exog_long.shape[1]
        self.k_precision = exog_precision.shape[1]
        self.k_extra = self.k_precision
        # Override nparams set by parent
        self._nparams = self.k_mean + self.k_precision
        self.df_model = self.k_mean + self.k_precision - 2
        self.df_resid = self.nobs - self.k_mean - self.k_precision
        self.hess_type = "oim"

        if "exog_precision" not in self._init_keys:
            self._init_keys.extend(["exog_precision"])
        self._init_keys.extend(["link", "link_precision", "group_idx"])
        self._null_drop_keys = ["exog_precision"]

        del kwds["extra_params_names"]
        self._check_kwargs(kwds)
        self.results_class = EcologicalBetaResults
        self.results_class_wrapper = EcologicalBetaResultsWrapper

    def _build_aggregation_matrix(self):
        """Build sparse-like aggregation structure for efficient computation."""
        # Store indices for each region
        self._region_subunit_indices = []
        for i in range(self.n_regions):
            indices = np.where(self.group_idx == i)[0]
            self._region_subunit_indices.append(indices)

    def _aggregate_to_regions(self, subunit_values):
        """Aggregate subunit-level values to region level using normalized weights."""
        region_values = np.zeros(self.n_regions)
        for i in range(self.n_regions):
            indices = self._region_subunit_indices[i]
            if len(indices) > 0:
                # Weighted sum (weights are already normalized to sum to 1)
                region_values[i] = np.sum(
                    self.weights[indices] * subunit_values[indices]
                )
        return region_values

    def _broadcast_to_subunits(self, region_values):
        """Broadcast region-level values to subunit level."""
        return region_values[self.group_idx]

    @property
    def nparams(self):
        return self._nparams

    @nparams.setter
    def nparams(self, value):
        # Allow parent class to set, but we'll override with our value
        pass

    def _split_params(self, params):
        """Split parameter vector into mean and precision parameters."""
        params_mean = params[: self.k_mean]
        params_prec = params[self.k_mean :]
        return params_mean, params_prec

    def _compute_mu_and_components(self, params):
        """Compute mu and intermediate values needed for likelihood and derivatives.

        Returns
        -------
        dict with keys:
            eta_long : linear predictor at subunit level
            mu_long : inverse-link at subunit level
            mu : aggregated mean at region level
            phi : precision at region level
            alpha, beta : Beta distribution parameters
        """
        params_mean, params_prec = self._split_params(params)

        # Subunit-level linear predictor and inverse link
        eta_long = self.exog_long @ params_mean
        mu_long = self.link.inverse(eta_long)

        # Aggregate to region level
        mu = self._aggregate_to_regions(mu_long)

        # Precision (region-level)
        linpred_prec = self.exog_precision @ params_prec
        phi = self.link_precision.inverse(linpred_prec)

        # Beta distribution parameters with numerical safeguards
        eps_lb = 1e-200
        alpha = np.clip(mu * phi, eps_lb, np.inf)
        beta = np.clip((1 - mu) * phi, eps_lb, np.inf)

        return {
            "eta_long": eta_long,
            "mu_long": mu_long,
            "mu": mu,
            "phi": phi,
            "alpha": alpha,
            "beta": beta,
            "linpred_prec": linpred_prec,
        }

    def loglikeobs(self, params):
        """Log-likelihood for each region (observation).

        Parameters
        ----------
        params : ndarray
            Model parameters [beta, precision_params].

        Returns
        -------
        ll : ndarray
            Log-likelihood for each region.
        """
        comp = self._compute_mu_and_components(params)
        y = self.endog
        phi = comp["phi"]
        alpha = comp["alpha"]
        beta = comp["beta"]

        ll = (
            lgamma(phi)
            - lgamma(alpha)
            - lgamma(beta)
            + (alpha - 1) * np.log(y)
            + (beta - 1) * np.log(1 - y)
        )
        return ll

    def score(self, params):
        """Score vector (gradient of log-likelihood).

        The score for beta_k is:
            s_{β_k} = Σ_i (∂L_i/∂μ_i) (∂μ_i/∂β_k)

        where:
            ∂L_i/∂μ_i = φ (y*_i - ψ(μ_i φ) + ψ((1-μ_i)φ))
            ∂μ_i/∂β_k = Σ_w g'^{-1}(g^{-1}(η_{iw})) x_{iwk}
                      = Σ_w link.inverse_deriv(η_{iw}) x_{iwk}

        Parameters
        ----------
        params : ndarray
            Parameter vector.

        Returns
        -------
        score : ndarray
            Gradient vector.
        """
        comp = self._compute_mu_and_components(params)
        y = self.endog
        mu = comp["mu"]
        phi = comp["phi"]
        eta_long = comp["eta_long"]

        # ∂L_i/∂μ_i at region level (Equation from Ferrari & Cribari-Neto)
        ystar = np.log(y / (1 - y))
        mustar = digamma(mu * phi) - digamma((1 - mu) * phi)
        dL_dmu = phi * (ystar - mustar)  # shape (n_regions,)

        # Broadcast to subunit level
        dL_dmu_long = self._broadcast_to_subunits(dL_dmu)

        # ∂μ_i/∂η_{iw} = weights_w · link.inverse_deriv(η_{iw})
        dmu_deta = self.weights * self.link.inverse_deriv(
            eta_long
        )  # shape (n_subunits,)

        # Score for mean parameters: X_long^T @ (dL_dmu_long * dmu_deta)
        score_beta = self.exog_long.T @ (dL_dmu_long * dmu_deta)

        # Score for precision parameters (standard beta regression formula)
        yt = np.log(1 - y)
        mut = digamma((1 - mu) * phi) - digamma(phi)
        h = 1.0 / self.link_precision.deriv(phi)
        sf2 = h * (mu * (ystar - mustar) + yt - mut)
        score_phi = self.exog_precision.T @ sf2

        return np.concatenate([score_beta, score_phi])

    def hessian(self, params, observed=None):
        """Hessian matrix (second derivative of log-likelihood).

        For the ecological model, the Fisher information for β is:

            I_{β_k β_l} = Σ_i (∂²L_i/∂μ_i²) (∂μ_i/∂β_k)(∂μ_i/∂β_l)

        where:
            ∂²L_i/∂μ_i² = -φ² (ψ'(μ_i φ) + ψ'((1-μ_i)φ))
            ∂μ_i/∂β_k = Σ_w link.inverse_deriv(η_{iw}) x_{iwk}

        Parameters
        ----------
        params : ndarray
            Parameter vector.
        observed : bool, optional
            If True, return observed Hessian. If False, return expected (Fisher) information.
            Default uses self.hess_type.

        Returns
        -------
        hess : ndarray
            Hessian matrix.
        """
        if observed is None:
            observed = self.hess_type == "oim"

        comp = self._compute_mu_and_components(params)
        y = self.endog
        mu = comp["mu"]
        phi = comp["phi"]
        eta_long = comp["eta_long"]

        # ∂²L_i/∂μ_i² (negative of this is the weight for Fisher information)
        d2L_dmu2 = -(phi**2) * (trigamma(mu * phi) + trigamma((1 - mu) * phi))

        # For the mean parameters, we need the Jacobian J where row i is:
        # J_i = Σ_{w ∈ region_i} weights_w · link.inverse_deriv(η_{iw}) · x_{iw}
        dmu_deta = self.weights * self.link.inverse_deriv(eta_long)  # (n_subunits,)

        # Build Jacobian matrix J (n_regions x k_mean)
        J = np.zeros((self.n_regions, self.k_mean))
        for i in range(self.n_regions):
            indices = self._region_subunit_indices[i]
            if len(indices) > 0:
                # J[i] = Σ_w weights_w * dmu_deta[w] * x_w
                J[i] = (dmu_deta[indices, None] * self.exog_long[indices]).sum(axis=0)

        # Hessian for beta-beta block: J^T @ diag(d2L_dmu2) @ J
        H_bb = J.T @ (d2L_dmu2[:, None] * J)

        # For observed Hessian, we need additional terms from chain rule
        if observed:
            # Additional term from second derivative of link
            ystar = np.log(y / (1 - y))
            mustar = digamma(mu * phi) - digamma((1 - mu) * phi)
            dL_dmu = phi * (ystar - mustar)

            # For each subunit, add contribution from link second derivative
            # d²μ/dη² = weights_w · link.inverse_deriv2(η)
            # This contributes: Σ_i (dL/dμ_i) Σ_w weights_w · link.inverse_deriv2(η_{iw}) x_{iw} x_{iw}^T
            dL_dmu_long = self._broadcast_to_subunits(dL_dmu)
            d2mu_deta2 = self.weights * self.link.inverse_deriv2(eta_long)

            # Add diagonal contribution (approximate, treating cross-terms as zero)
            for w in range(self.n_subunits):
                H_bb += (
                    dL_dmu_long[w]
                    * d2mu_deta2[w]
                    * np.outer(self.exog_long[w], self.exog_long[w])
                )

        # Precision block (same as standard beta regression)
        h = 1.0 / self.link_precision.deriv(phi)
        ystar = np.log(y / (1 - y))
        mustar = digamma(mu * phi) - digamma((1 - mu) * phi)
        yt = np.log(1 - y)
        mut = digamma((1 - mu) * phi) - digamma(phi)
        ymu_star = ystar - mustar

        trig_alpha = trigamma(mu * phi)
        trig_beta = trigamma((1 - mu) * phi)
        trig_phi = trigamma(phi)

        var_star = trig_alpha + trig_beta
        var_t = trig_beta - trig_phi
        c = -trig_beta

        # H_φφ
        jgg = h**2 * (mu**2 * var_star + 2 * mu * c + var_t)
        if observed:
            q = self.link_precision.deriv2(phi)
            jgg += (mu * ymu_star + yt - mut) * q * h**3

        H_pp = -(self.exog_precision.T * jgg) @ self.exog_precision

        # Cross term H_βφ
        # ∂²L/(∂β_k ∂φ) = Σ_i (∂²L_i/∂μ_i∂φ)(∂μ_i/∂β_k) + (∂L_i/∂μ_i)(∂²μ_i/∂β_k∂φ)
        # The second term is zero since μ doesn't depend on φ
        # ∂²L_i/∂μ_i∂φ = (ystar - mustar) - φ(μψ'(μφ) - (1-μ)ψ'((1-μ)φ))

        d2L_dmu_dphi = (ystar - mustar) - phi * (mu * trig_alpha - (1 - mu) * trig_beta)

        # Also need ∂φ/∂params_prec
        # ∂L/∂params_prec = (∂L/∂φ)(∂φ/∂η_prec)(∂η_prec/∂params_prec)
        # So ∂²L/(∂β_k ∂params_prec) = (∂²L/∂μ∂φ)(∂μ/∂β_k)(∂φ/∂η_prec)(∂η_prec/∂params_prec)
        #                           = J^T @ diag(d2L_dmu_dphi * h) @ Z

        H_bp = J.T @ (d2L_dmu_dphi[:, None] * h[:, None] * self.exog_precision)

        # Assemble full Hessian
        H = np.block([[H_bb, H_bp], [H_bp.T, H_pp]])

        return H

    def _start_params(self, niter=2):
        """Compute starting parameters using standard Beta regression on aggregated data.

        Parameters
        ----------
        niter : int
             Ignored. Kept for API compatibility.

        Returns
        -------
        start_params : ndarray
            Initial parameter estimates.
        """
        y = self.endog

        # For initial beta: use region-level average of covariates
        X_region = np.zeros((self.n_regions, self.k_mean))
        for i in range(self.n_regions):
            indices = self._region_subunit_indices[i]
            if len(indices) > 0:
                X_region[i] = self.exog_long[indices].mean(axis=0)

        # Use statsmodels BetaModel for starting values
        mod = BetaModel(
            y,
            X_region,
            exog_precision=self.exog_precision,
            link=self.link,
            link_precision=self.link_precision,
        )
        # Use bfgs for robustness, though default 'newton' is fine too
        # disp=0 to silence convergence messages for start_params
        res = mod.fit(disp=0)

        return res.params

    def fit(self, start_params=None, maxiter=1000, disp=False, method="bfgs", **kwds):
        """Fit the model by maximum likelihood.

        Parameters
        ----------
        start_params : array-like, optional
            Starting values for optimization. If None, computed automatically.
        maxiter : int
            Maximum number of iterations.
        disp : bool
            Display convergence information.
        method : str
            Optimization method.
        **kwds :
            Additional keyword arguments for optimizer.

        Returns
        -------
        EcologicalBetaResults
            Results instance.
        """
        if start_params is None:
            start_params = self._start_params()

        if "cov_type" in kwds:
            if kwds["cov_type"].lower() == "eim":
                self.hess_type = "eim"
                del kwds["cov_type"]
        else:
            self.hess_type = "oim"

        res = super().fit(
            start_params=start_params, maxiter=maxiter, method=method, disp=disp, **kwds
        )

        if not isinstance(res, EcologicalBetaResultsWrapper):
            res = EcologicalBetaResultsWrapper(res)

        return res

    def predict(self, params, exog=None, exog_precision=None, which="mean"):
        """Predict values for mean or precision.

        Parameters
        ----------
        params : array_like
            Model parameters.
        exog : array_like, optional
            Subunit-level covariates. If None, uses training data.
        exog_precision : array_like, optional
            Region-level precision covariates. If None, uses training data.
        which : str
            What to predict:
            - "mean" : aggregated mean at region level
            - "mean_subunit" : mean at subunit level (before aggregation)
            - "precision" : precision parameter
            - "linear" : linear predictor at subunit level
            - "linear-precision" : linear predictor for precision

        Returns
        -------
        predicted : ndarray
            Predicted values.
        """
        params_mean, params_prec = self._split_params(params)

        if which in ["mean", "mean_subunit", "linear"]:
            if exog is None:
                exog = self.exog_long

            linpred = exog @ params_mean

            if which == "linear":
                return linpred
            elif which == "mean_subunit":
                return self.link.inverse(linpred)
            else:  # "mean"
                mu_long = self.link.inverse(linpred)
                return self._aggregate_to_regions(mu_long)

        elif which in ["precision", "linear-precision"]:
            if exog_precision is None:
                exog_precision = self.exog_precision

            linpred_prec = exog_precision @ params_prec

            if which == "linear-precision":
                return linpred_prec
            else:
                return self.link_precision.inverse(linpred_prec)

        else:
            raise ValueError(f"which='{which}' is not available")

    def predict_subunit(self, params, subunit_idx=None):
        """Predict mean for specific subunits.

        This is useful for inference at the woreda/zone level after fitting
        at the region level.

        Parameters
        ----------
        params : array_like
            Model parameters.
        subunit_idx : array_like, optional
            Indices of subunits to predict for. If None, predicts for all.

        Returns
        -------
        mu_subunit : ndarray
            Predicted mean for each subunit.
        """
        params_mean, _ = self._split_params(params)

        if subunit_idx is None:
            exog = self.exog_long
        else:
            exog = self.exog_long[subunit_idx]

        eta = exog @ params_mean
        return self.link.inverse(eta)


class EcologicalBetaResults(GenericLikelihoodModelResults, _LLRMixin):
    """Results class for Ecological Beta regression."""

    @cache_readonly
    def fittedvalues(self):
        """In-sample predicted mean (region-level aggregated)."""
        return self.model.predict(self.params, which="mean")

    @cache_readonly
    def fittedvalues_subunit(self):
        """In-sample predicted mean at subunit level."""
        return self.model.predict(self.params, which="mean_subunit")

    @cache_readonly
    def fitted_precision(self):
        """In-sample predicted precision."""
        return self.model.predict(self.params, which="precision")

    @cache_readonly
    def resid(self):
        """Response residual (region-level)."""
        return self.model.endog - self.fittedvalues

    @cache_readonly
    def resid_pearson(self):
        """Pearson standardized residual."""
        mu = self.fittedvalues
        phi = self.fitted_precision
        var = mu * (1 - mu) / (1 + phi)
        return self.resid / np.sqrt(var)

    @cache_readonly
    def prsquared(self):
        """Variance-explained pseudo-R-squared.

        Computed as 1 - SS_res / SS_tot on the region-level predictions.
        This is equivalent to the coefficient of determination and works
        correctly regardless of log-likelihood sign.

        Note: We compute this directly rather than using the parent's
        pseudo_rsquared method because our model requires group_idx which
        breaks the standard null model construction.
        """
        y = self.model.endog
        y_pred = self.fittedvalues

        # Total sum of squares
        ss_tot = np.sum((y - y.mean()) ** 2)

        # Residual sum of squares
        ss_res = np.sum((y - y_pred) ** 2)

        if ss_tot > 0:
            return 1 - ss_res / ss_tot
        else:
            return 0.0

    def predict_subunit(self, subunit_idx=None):
        """Predict mean for subunits (woreda/zone level inference).

        Parameters
        ----------
        subunit_idx : array_like, optional
            Indices of subunits to predict. If None, predicts for all.

        Returns
        -------
        mu : ndarray
            Predicted mean for each subunit.
        """
        return self.model.predict_subunit(self.params, subunit_idx)

    def get_distribution_params(self, exog=None, exog_precision=None):
        """Return Beta distribution parameters (alpha, beta).

        Parameters
        ----------
        exog : array_like, optional
            Subunit-level covariates.
        exog_precision : array_like, optional
            Precision covariates.

        Returns
        -------
        alpha, beta : tuple of ndarrays
            Beta distribution shape parameters.
        """
        mean = self.model.predict(self.params, exog=exog, which="mean")
        precision = self.model.predict(
            self.params, exog_precision=exog_precision, which="precision"
        )
        return precision * mean, precision * (1 - mean)

    def get_distribution(self, exog=None, exog_precision=None):
        """Return scipy Beta distribution instance.

        Parameters
        ----------
        exog : array_like, optional
            Subunit-level covariates.
        exog_precision : array_like, optional
            Precision covariates.

        Returns
        -------
        distr : scipy.stats.beta
            Frozen Beta distribution.
        """
        from scipy import stats

        alpha, beta = self.get_distribution_params(exog, exog_precision)
        return stats.beta(alpha, beta)


class EcologicalBetaResultsWrapper(lm.RegressionResultsWrapper):
    pass


wrap.populate_wrapper(EcologicalBetaResultsWrapper, EcologicalBetaResults)


if __name__ == "__main__":
    # Simple test with synthetic data
    np.random.seed(42)

    # Generate synthetic data
    n_regions = 100
    n_subunits_per_region = 5
    n_subunits = n_regions * n_subunits_per_region
    k_mean = 3  # number of covariates

    # Create group indices
    group_idx = np.repeat(np.arange(n_regions), n_subunits_per_region)

    # Generate subunit-level covariates
    X_long = np.random.randn(n_subunits, k_mean)
    X_long[:, 0] = 1  # intercept

    # True parameters
    beta_true = np.array([0.0, -0.5, 0.3])  # intercept at 0 for centered mean
    phi_true = 50.0

    # Generate true mu at subunit level
    eta_long = X_long @ beta_true
    mu_long = LOGIT_LINK.inverse(eta_long)

    # Aggregate to region level using equal weights (average)
    mu_region = np.zeros(n_regions)
    for i in range(n_regions):
        mask = group_idx == i
        mu_region[i] = mu_long[mask].mean()  # weighted average with equal weights

    # Generate beta-distributed response
    from scipy import stats

    alpha = mu_region * phi_true
    beta_param = (1 - mu_region) * phi_true
    y = stats.beta.rvs(alpha, beta_param)

    # Fit the ecological beta model
    print("Fitting Ecological Beta Model...")
    print(f"True parameters: beta = {beta_true}, phi = {phi_true}")
    print(f"Response range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"True mu range: [{mu_region.min():.4f}, {mu_region.max():.4f}]")

    model = EcologicalBetaModel(y, X_long, group_idx)
    result = model.fit(disp=True)

    print("\nFitted parameters:")
    print(result.summary())

    print("\nParameter comparison:")
    print(f"  beta estimates: {result.params[:k_mean]}")
    print(f"  beta true:      {beta_true}")
    print(f"  phi estimate:   {np.exp(result.params[-1]):.2f}")
    print(f"  phi true:       {phi_true}")

    # Test subunit prediction
    subunit_predictions = result.predict_subunit()
    print(f"\nSubunit predictions shape: {subunit_predictions.shape}")
    print(
        f"Subunit predictions range: [{subunit_predictions.min():.4f}, {subunit_predictions.max():.4f}]"
    )
    print(f"True subunit mu range: [{mu_long.min():.4f}, {mu_long.max():.4f}]")
