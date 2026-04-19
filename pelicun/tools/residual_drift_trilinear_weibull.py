"""
RID Project
RID|PID models
"""

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
from scipy.optimize import basinhopping, minimize
from scipy.special import erfc, erfcinv

np.set_printoptions(formatter={'float': '{:0.5f}'.format})


class Model:
    """
    Base class for conditional models with common functionality.
    """

    def __init__(self) -> None:
        self.raw_pid: npt.NDArray | None = None
        self.raw_rid: npt.NDArray | None = None

        self.uniform_sample: npt.NDArray | None = None
        self.sim_pid: npt.NDArray | None = None
        self.sim_rid: npt.NDArray | None = None

        self.censoring_limit: float | None = None
        self.parameters: npt.NDArray | None = None
        self.parameter_names: list[str] | None = None
        self.parameter_bounds: list[tuple[float, float]] | None = None
        self.fit_status = False
        self.fit_meta: Any = None

        self.rolling_pid: npt.NDArray | None = None
        self.rolling_rid_50: npt.NDArray | None = None
        self.rolling_rid_20: npt.NDArray | None = None
        self.rolling_rid_80: npt.NDArray | None = None

    def add_data(self, raw_pid: npt.NDArray, raw_rid: npt.NDArray) -> None:
        self.raw_pid = raw_pid
        self.raw_rid = raw_rid

    def calculate_rolling_quantiles(
        self,
        fraction: float = 0.075,
        min_points: int = 5,
    ) -> None:
        assert self.raw_pid is not None
        assert self.raw_rid is not None

        idsort = np.argsort(self.raw_pid)
        pid_vals_sorted = self.raw_pid[idsort]
        rid_vals_sorted = self.raw_rid[idsort]

        num_vals = len(pid_vals_sorted)

        group_size = max(int(num_vals * fraction), min_points)

        rolling_pid = np.array(
            [
                np.mean(pid_vals_sorted[i : i + group_size])
                for i in range(num_vals - group_size + 1)
            ]
        )

        rolling_rid_50 = np.array(
            [
                np.quantile(rid_vals_sorted[i : i + group_size], 0.50)
                for i in range(num_vals - group_size + 1)
            ]
        )

        rolling_rid_20 = np.array(
            [
                np.quantile(rid_vals_sorted[i : i + group_size], 0.20)
                for i in range(num_vals - group_size + 1)
            ]
        )

        rolling_rid_80 = np.array(
            [
                np.quantile(rid_vals_sorted[i : i + group_size], 0.80)
                for i in range(num_vals - group_size + 1)
            ]
        )

        self.rolling_pid = rolling_pid
        self.rolling_rid_50 = rolling_rid_50
        self.rolling_rid_20 = rolling_rid_20
        self.rolling_rid_80 = rolling_rid_80

    def fit(
        self,
        method: Literal['mle', 'quantiles', 'mle-fast'] = 'mle',
        *,
        global_search: bool = True,
    ) -> None:
        # Initial values

        if method == 'quantiles':
            use_objective = self.get_quantile_objective
            jac = None
            optimizer_method = 'Nelder-Mead'
        elif method == 'mle':
            use_objective = self.get_mle_objective
            jac = None
            optimizer_method = 'Nelder-Mead'
        elif method == 'mle-fast':
            use_objective = self.get_mle_objective_fast
            jac = True
            optimizer_method = 'L-BFGS-B'
        else:
            raise ValueError(f'Invalid method: `{method}`.')  # noqa: EM102, TRY003

        if global_search:
            # Basinhopping for global search
            result = basinhopping(
                func=use_objective,
                x0=self.parameters,
                niter=20,
                T=1.0,
                stepsize=0.02,
                niter_success=5,
                minimizer_kwargs={
                    'method': optimizer_method,
                    'jac': jac,
                    'bounds': self.parameter_bounds,
                    'options': {'maxiter': 200},
                    'tol': 1e-6,
                },
            )
            # result = basinhopping(
            #     func=use_objective,
            #     x0=self.parameters,
            #     T=1000.00,
            #     minimizer_kwargs={
            #         'method': optimizer_method,
            #         'jac': jac,
            #         'bounds': self.parameter_bounds,
            #         'options': {'maxiter': 1000},
            #         'tol': 1e-6,
            #     },
            # )
        else:
            n_repeats = 3
            success = False
            while (not success) and n_repeats > 0:
                # Single step optimization (local minimum)
                result = minimize(
                    fun=use_objective,
                    x0=self.parameters,
                    jac=jac,
                    bounds=self.parameter_bounds,
                    method=optimizer_method,
                    options={'maxiter': 10000},
                    tol=1e-10,
                )
                success = result.success
                n_repeats -= 1

        self.fit_meta = result
        assert result.success, 'Minimization failed.'
        self.parameters = result.x

    def get_mle_objective(self, parameters: npt.NDArray) -> float:
        # update the parameters
        self.parameters = parameters
        density = self.evaluate_pdf(self.raw_rid, self.raw_pid, self.censoring_limit)
        negloglikelihood = -np.sum(np.log(density))
        return negloglikelihood  # noqa: RET504

    def get_quantile_objective(self, parameters: npt.NDArray) -> float:
        # update the parameters
        self.parameters = parameters

        # calculate the model's RID|PID quantiles
        if self.rolling_pid is None:
            self.calculate_rolling_quantiles()
        model_pid = self.rolling_pid
        model_rid_50 = self.evaluate_inverse_cdf(0.50, model_pid)
        model_rid_20 = self.evaluate_inverse_cdf(0.20, model_pid)
        model_rid_80 = self.evaluate_inverse_cdf(0.80, model_pid)

        loss = (
            (self.rolling_rid_50 - model_rid_50).T
            @ (self.rolling_rid_50 - model_rid_50)
            + (self.rolling_rid_20 - model_rid_20).T
            @ (self.rolling_rid_20 - model_rid_20)
            + (self.rolling_rid_80 - model_rid_80).T
            @ (self.rolling_rid_80 - model_rid_80)
        )

        return loss  # noqa: RET504

    def generate_rid_samples(self, pid_samples: npt.NDArray) -> npt.NDArray:
        if self.uniform_sample is None:
            self.uniform_sample = np.random.uniform(0.00, 1.00, len(pid_samples))

        rid_samples = self.evaluate_inverse_cdf(self.uniform_sample, pid_samples)

        self.sim_pid = pid_samples
        self.sim_rid = rid_samples

        return rid_samples

    def plot_data(self, ax=None, scatter_kwargs=None) -> None:  # noqa: ANN001
        """
        Add a scatter plot of the raw data to a matplotlib axis, or
        show it if one is not given.
        """
        if scatter_kwargs is None:
            scatter_kwargs = {
                's': 5.0,
                'facecolor': 'none',
                'edgecolor': 'black',
                'alpha': 0.2,
            }

        if ax is None:
            _, ax = plt.subplots()
            ax.scatter(self.raw_pid, self.raw_rid, **scatter_kwargs)
            plt.show()
        else:
            ax.scatter(self.raw_pid, self.raw_rid, **scatter_kwargs)

    def plot_model(
        self,
        ax,  # noqa: ANN001
        rolling=True,  # noqa: ANN001, FBT002
        training=True,  # noqa: ANN001, FBT002
        model=True,  # noqa: ANN001, FBT002
        model_color='C0',  # noqa: ANN001, FBT002, RUF100
    ) -> None:
        """
        Plot the data in a scatter plot,
        superimpose their empirical quantiles,
        and the quantiles resulting from the fitted model.
        """
        if self.fit_status == 'False':
            self.fit()

        if training:
            self.plot_data(ax)

        if rolling:
            self.calculate_rolling_quantiles()

            ax.plot(self.rolling_pid, self.rolling_rid_50, 'k')
            ax.plot(self.rolling_pid, self.rolling_rid_20, 'k', linestyle='dashed')
            ax.plot(self.rolling_pid, self.rolling_rid_80, 'k', linestyle='dashed')

        if model:
            # model_pid = np.linspace(0.00, self.rolling_pid[-1], 1000)
            model_pid = np.linspace(0.00, 0.08, 1000)
            model_rid_50 = self.evaluate_inverse_cdf(0.50, model_pid)
            model_rid_20 = self.evaluate_inverse_cdf(0.20, model_pid)
            model_rid_80 = self.evaluate_inverse_cdf(0.80, model_pid)

            ax.plot(model_pid, model_rid_50, model_color)
            ax.plot(model_pid, model_rid_20, model_color, linestyle='dashed')
            ax.plot(model_pid, model_rid_80, model_color, linestyle='dashed')

    def evaluate_inverse_cdf(self, quantile, pid):  # noqa: ANN001, ANN201
        """
        Evaluate the inverse of the conditional RID|PID CDF.
        """
        raise NotImplementedError('Subclasses should implement this.')  # noqa: EM101

    def evaluate_pdf(self, rid, pid, censoring_limit=None):  # noqa: ANN001, ANN201
        """
        Evaluate the conditional RID|PID PDF.
        """
        raise NotImplementedError('Subclasses should implement this.')  # noqa: EM101

    def evaluate_cdf(self, rid, pid):  # noqa: ANN001, ANN201
        """
        Evaluate the conditional RID|PID CDF.
        """
        raise NotImplementedError('Subclasses should implement this.')  # noqa: EM101

    def trilinear_fnc(self, pid: npt.NDArray, y0, m0, m1, m2, x0, x1) -> npt.NDArray:  # noqa: ANN001, PLR6301
        y1 = y0 + m0 * x0
        y2 = y1 + m1 * (x1 - x0)
        res = m0 * pid + y0
        mask = pid > x0
        res[mask] = (pid[mask] - x0) * m1 + y1
        mask = pid > x1
        res[mask] = (pid[mask] - x1) * m2 + y2
        return res


class Model_Trilinear_Weibull(Model):
    """
    Weibull model
    """

    def __init__(self) -> None:
        super().__init__()
        # initial parameters
        self.parameters = np.array((0.005, 0.15, 0.020 - 0.005, 0.25, 1.20, 1.20))
        # parameter names
        self.parameter_names = [
            'pid_0',
            'lambda_slope_1',
            'pid_1_minus_pid_0',
            'lambda_slope_2_minus_lambda_slope_1',
            'kappa_1',
            'kappa_2',
        ]
        # bounds
        self.parameter_bounds = [
            (1e-4, 0.05),
            (0.00, 1.00),
            (1e-3, 0.1),
            (0.00, 1.00),
            (1.0, 10.0),
            (1.0, 10.0),
        ]

    def obtain_lambda(
        self,
        pid: npt.NDArray,
    ) -> npt.NDArray:
        assert self.parameters is not None
        (
            pid_0,
            lambda_slope_1,
            pid_1_minus_pid_0,
            lambda_slope_2_minus_lambda_slope_1,
            _kappa_1,
            _kappa_2,
        ) = self.parameters
        lambda_slope_2 = lambda_slope_1 + lambda_slope_2_minus_lambda_slope_1
        pid_1 = pid_0 + pid_1_minus_pid_0
        lambda_0 = 1e-6
        lambda_1 = lambda_0 + lambda_slope_1 * pid_1_minus_pid_0
        lambda_trilinear = np.empty_like(pid)
        mask = pid <= pid_0
        lambda_trilinear[mask] = 1e-6
        mask = np.logical_and(pid_0 < pid, pid <= pid_1)
        lambda_trilinear[mask] = (pid[mask] - pid_0) * lambda_slope_1 + lambda_0
        mask = pid_1 < pid
        lambda_trilinear[mask] = (pid[mask] - pid_1) * lambda_slope_2 + lambda_1
        return lambda_trilinear

    def obtain_kappa(
        self,
        pid: npt.NDArray,
    ) -> npt.NDArray:
        assert self.parameters is not None
        (
            pid_0,
            _lambda_slope_1,
            pid_1_minus_pid_0,
            _lambda_slope_2_minus_lambda_slope_1,
            kappa_1,
            kappa_2,
        ) = self.parameters
        # calculate slope for kappa
        kappa_slope = (kappa_2 - kappa_1) / pid_1_minus_pid_0
        pid_1 = pid_0 + pid_1_minus_pid_0
        kappa_trilinear = np.empty_like(pid)
        mask = pid <= pid_0
        kappa_trilinear[mask] = kappa_1
        mask = np.logical_and(pid_0 < pid, pid <= pid_1)
        kappa_trilinear[mask] = (pid[mask] - pid_0) * kappa_slope + kappa_1
        mask = pid_1 < pid
        kappa_trilinear[mask] = kappa_2
        return kappa_trilinear

    def obtain_lambda_grad(
        self,
        pid: npt.NDArray,
    ) -> npt.NDArray:
        assert self.parameters is not None
        (
            pid_0,
            lambda_slope_1,
            pid_1_minus_pid_0,
            lambda_slope_2_minus_lambda_slope_1,
            _kappa_1,
            _kappa_2,
        ) = self.parameters
        lambda_slope_2 = lambda_slope_1 + lambda_slope_2_minus_lambda_slope_1
        lambda_trilinear = np.empty((len(pid), 6))

        mask = pid <= pid_0
        lambda_trilinear[mask, 0] = 0.00
        lambda_trilinear[mask, 1] = 0.00
        lambda_trilinear[mask, 2] = 0.00
        lambda_trilinear[mask, 3] = 0.00
        lambda_trilinear[mask, 4] = 0.00
        lambda_trilinear[mask, 5] = 0.00

        mask = np.logical_and(pid_0 < pid, pid <= pid_0 + pid_1_minus_pid_0)
        lambda_trilinear[mask, 0] = -lambda_slope_1
        lambda_trilinear[mask, 1] = pid[mask] - pid_0
        lambda_trilinear[mask, 2] = 0.00
        lambda_trilinear[mask, 3] = 0.00
        lambda_trilinear[mask, 4] = 0.00
        lambda_trilinear[mask, 5] = 0.00

        mask = pid_0 + pid_1_minus_pid_0 < pid
        lambda_trilinear[mask, 0] = -lambda_slope_2
        lambda_trilinear[mask, 1] = pid[mask] - pid_0
        lambda_trilinear[mask, 2] = lambda_slope_1 - lambda_slope_2
        lambda_trilinear[mask, 3] = pid[mask] - (pid_0 + pid_1_minus_pid_0)
        lambda_trilinear[mask, 4] = 0.00
        lambda_trilinear[mask, 5] = 0.00

        return lambda_trilinear

    def obtain_kappa_grad(
        self,
        pid: npt.NDArray,
    ) -> npt.NDArray:
        assert self.parameters is not None
        (
            pid_0,
            _lambda_slope_1,
            pid_1_minus_pid_0,
            _lambda_slope_2_minus_lambda_slope_1,
            kappa_1,
            kappa_2,
        ) = self.parameters
        kappa_trilinear = np.empty((len(pid), 6))

        mask = pid <= pid_0
        kappa_trilinear[mask, 0] = 0.00
        kappa_trilinear[mask, 1] = 0.00
        kappa_trilinear[mask, 2] = 0.00
        kappa_trilinear[mask, 3] = 0.00
        kappa_trilinear[mask, 4] = 1.00
        kappa_trilinear[mask, 5] = 0.00

        mask = np.logical_and(pid_0 < pid, pid <= pid_0 + pid_1_minus_pid_0)
        kappa_trilinear[mask, 0] = -(kappa_2 - kappa_1) / pid_1_minus_pid_0
        kappa_trilinear[mask, 1] = 0.00
        kappa_trilinear[mask, 2] = (
            -(pid[mask] - pid_0) * (kappa_2 - kappa_1) / pid_1_minus_pid_0**2
        )
        kappa_trilinear[mask, 3] = 0.00
        kappa_trilinear[mask, 4] = -((pid[mask] - pid_0) / pid_1_minus_pid_0) + 1.00
        kappa_trilinear[mask, 5] = (pid[mask] - pid_0) / pid_1_minus_pid_0

        mask = pid_0 + pid_1_minus_pid_0 < pid
        kappa_trilinear[mask, 0] = 0.00
        kappa_trilinear[mask, 1] = 0.00
        kappa_trilinear[mask, 2] = 0.00
        kappa_trilinear[mask, 3] = 0.00
        kappa_trilinear[mask, 4] = 0.00
        kappa_trilinear[mask, 5] = 1.00

        return kappa_trilinear

    def get_mle_objective_fast(
        self, parameters: npt.NDArray
    ) -> tuple[float, npt.NDArray]:
        """
        Compute the negative log-likelihood and its gradient.

        Compute negative log-likelihood and gradient using a
        numerically stable formulation.

        Parameters
        ----------
        parameters : npt.NDArray
            Model parameters.

        Returns
        -------
        tuple[float, npt.NDArray]
            Objective value and gradient vector.
        """
        self.parameters = parameters

        c_lim = float(self.censoring_limit)
        if c_lim <= 0.0:
            msg = 'censoring_limit must be positive for the stable censored formulation.'
            raise ValueError(msg)

        mask_censored = self.raw_rid < c_lim
        mask_uncensored = ~mask_censored

        # Model values
        lambda_vec = self.obtain_lambda(self.raw_pid)[:, np.newaxis]
        kappa_vec = self.obtain_kappa(self.raw_pid)[:, np.newaxis]
        rid_vec = self.raw_rid[:, np.newaxis]

        lambda_grad = self.obtain_lambda_grad(self.raw_pid)
        kappa_grad = self.obtain_kappa_grad(self.raw_pid)

        # Numerical safeguards for log/division operations
        tiny = 1e-300
        lambda_safe = np.maximum(lambda_vec, tiny)
        kappa_safe = np.maximum(kappa_vec, tiny)

        n_obs = len(self.raw_rid)
        n_params = len(self.parameters)
        grad_log_l = np.zeros((n_obs, n_params), dtype=float)
        log_l = np.zeros((n_obs, 1), dtype=float)

        # Exponentiation limits for float64
        max_log = 700.0
        min_log = -745.0

        # Uncensored observations
        if np.any(mask_uncensored):
            rid_u = rid_vec[mask_uncensored, :]
            lam_u = lambda_safe[mask_uncensored, :]
            kap_u = kappa_safe[mask_uncensored, :]
            lam_grad_u = lambda_grad[mask_uncensored, :]
            kap_grad_u = kappa_grad[mask_uncensored, :]

            rid_u_safe = np.maximum(rid_u, tiny)

            log_frac_u = np.log(rid_u_safe) - np.log(lam_u)
            log_t_u = kap_u * log_frac_u
            t_u = np.exp(np.clip(log_t_u, min_log, max_log))

            log_l[mask_uncensored, :] = (
                np.log(kap_u) - np.log(lam_u) + (kap_u - 1.0) * log_frac_u - t_u
            )

            grad_log_l[mask_uncensored, :] = (
                kap_grad_u / kap_u
                + kap_grad_u * log_frac_u * (1.0 - t_u)
                + lam_grad_u / lam_u * kap_u * (t_u - 1.0)
            )

        # Censored observations
        if np.any(mask_censored):
            lam_c = lambda_safe[mask_censored, :]
            kap_c = kappa_safe[mask_censored, :]
            lam_grad_c = lambda_grad[mask_censored, :]
            kap_grad_c = kappa_grad[mask_censored, :]

            log_c_over_l = np.log(c_lim) - np.log(lam_c)
            log_s_c = kap_c * log_c_over_l
            s_c = np.exp(np.clip(log_s_c, min_log, max_log))

            # stable evaluation of log(1 - exp(-s))
            log_l[mask_censored, :] = np.log(-np.expm1(-s_c))

            # stable evaluation of omega = s / (exp(s) - 1)
            omega_c = np.empty_like(s_c)
            small_mask = s_c < 1e-6
            large_mask = s_c > max_log

            # series expansion for small s
            if np.any(small_mask):
                s_small = s_c[small_mask]
                omega_c[small_mask] = (
                    1.0 - 0.5 * s_small + (s_small * s_small) / 12.0
                )

            if np.any(~small_mask & ~large_mask):
                s_mid = s_c[~small_mask & ~large_mask]
                omega_c[~small_mask & ~large_mask] = s_mid / np.expm1(s_mid)

            if np.any(large_mask):
                omega_c[large_mask] = 0.0

            grad_log_l[mask_censored, :] = omega_c * (
                log_c_over_l * kap_grad_c - (kap_c / lam_c) * lam_grad_c
            )

        # Likelihood floor in log-space
        log_l_floor = np.log(1e-6)

        clipped_mask = log_l[:, 0] < log_l_floor
        if np.any(clipped_mask):
            log_l[clipped_mask, :] = log_l_floor
            grad_log_l[clipped_mask, :] = 0.0

        objective = -float(np.sum(log_l))
        gradient = -np.sum(grad_log_l, axis=0)

        return objective, gradient

    def evaluate_pdf(
        self,
        rid: npt.NDArray,
        pid: npt.NDArray,
        censoring_limit: float | None = None,
    ) -> npt.NDArray:
        lambda_trilinear = self.obtain_lambda(pid)
        kappa_trilinear = self.obtain_kappa(pid)
        pdf_val = sp.stats.weibull_min.pdf(
            rid, kappa_trilinear, 0.00, lambda_trilinear
        )
        if censoring_limit:
            censored_range_mass = self.evaluate_cdf(
                np.full(len(pid), censoring_limit), pid
            )
            mask = rid <= censoring_limit
            pdf_val[mask] = censored_range_mass[mask]
        pdf_val[pdf_val < 1e-6] = 1e-6  # noqa: PLR2004
        return pdf_val

    def evaluate_cdf(self, rid: npt.NDArray, pid: npt.NDArray) -> npt.NDArray:
        lambda_trilinear = self.obtain_lambda(pid)
        kappa_trilinear = self.obtain_kappa(pid)
        return sp.stats.weibull_min.cdf(rid, kappa_trilinear, 0.00, lambda_trilinear)

    def evaluate_inverse_cdf(self, quantile: float, pid: npt.NDArray) -> npt.NDArray:
        lambda_trilinear = self.obtain_lambda(pid)
        kappa_trilinear = self.obtain_kappa(pid)
        return sp.stats.weibull_min.ppf(
            quantile, kappa_trilinear, 0.00, lambda_trilinear
        )


class ResidualDriftTrilinearWeibullModel:
    """
    Building-level residual drift model.

    Building-level residual drift model built from one or more
    `Model_Trilinear_Weibull` sub-models. Stores the residual drift
    inference configuration, fits the necessary conditional RID|PID
    models using structural analysis PSDR-RSDR data, stores the
    associated correlation structure, and is used to sample RID
    realizations for an entire building given statistically generated
    PID samples.

    Parameters
    ----------
    config: dict
        Configuration dictionary. Supported keys are:
        - `approach`: `'story'` or `'max_max'`
        - `fit_directions_together`: bool, optional. Default: False
        - `correlation_source`: `'model'` or a dict with keys
          `'c1'`, `'c2'`, and `'z_over_h'`.
        - `censoring_limit`: float, optional
        - `model_fit_method`: str, optional

    Notes
    -----
    Expected format of the structural analysis data passed to `fit`:
    - index: MultiIndex including at least `loc` and `dir`
    - columns: must include `PID` and `RID`

    Expected format of the PID sample passed to `sample`:
    - columns: MultiIndex with levels `edp_type`, `loc`, and `dir`
      after selecting the PID block
    """

    def __init__(self, config: dict) -> None:
        self.config = config.copy()

        self.approach = self.config['approach']
        self.fit_directions_together = self.config.get(
            'fit_directions_together', False
        )
        self.correlation_source = self.config.get('correlation_source', 'model')
        self.censoring_limit = float(self.config.get('censoring_limit', 0.00025))
        self.model_fit_method = self.config.get('model_fit_method', 'mle-fast')

        self.models_by_loc: dict[str, Model_Trilinear_Weibull] = {}
        self.models_by_loc_dir: dict[tuple[str, str], Model_Trilinear_Weibull] = {}
        self.model_maxmax: Model_Trilinear_Weibull | None = None
        self.models_maxmax_by_dir: dict[str, Model_Trilinear_Weibull] = {}

        self.locations: list[str] = []
        self.directions: list[str] = []
        self.correlation_matrix: pd.DataFrame | None = None
        self.fitted = False

        self._validate_config()

    def _validate_config(self) -> None:
        if self.approach not in {'story', 'max_max'}:
            msg = f'Invalid approach: `{self.approach}`.'
            raise ValueError(msg)

        if self.correlation_source != 'model':
            if not isinstance(self.correlation_source, dict):
                msg = (
                    '`correlation_source` must be `model` or a dictionary '
                    'defining the reference correlation model.'
                )
                raise ValueError(msg)

            required_keys = {'c1', 'c2', 'z_over_h'}
            missing_keys = required_keys - set(self.correlation_source)
            if missing_keys:
                msg = (
                    'Reference correlation configuration is incomplete. '
                    f'Missing: {sorted(missing_keys)}.'
                )
                raise ValueError(msg)

            if not isinstance(self.correlation_source['z_over_h'], dict):
                msg = '`z_over_h` must be a dictionary mapping location to z/H.'
                raise ValueError(msg)

    @staticmethod
    def _nearest_psd(mat: np.ndarray) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(mat)
        min_value = 1e-8
        eigvals[eigvals < min_value] = min_value
        out = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(out))
        out /= np.outer(d, d)
        np.fill_diagonal(out, 1.0)
        return out

    @staticmethod
    def _fit_single_model(
        raw_pid: np.ndarray,
        raw_rid: np.ndarray,
        censoring_limit: float,
        model_fit_method: str,
        global_search: bool = True,
    ) -> Model_Trilinear_Weibull:
        model = Model_Trilinear_Weibull()
        model.add_data(raw_pid=raw_pid, raw_rid=raw_rid)
        model.censoring_limit = censoring_limit
        model.fit(method=model_fit_method, global_search=global_search)
        return model

    def _estimate_reference_corr(self) -> pd.DataFrame:
        assert isinstance(self.correlation_source, dict)

        z_over_h = self.correlation_source['z_over_h']
        c_1 = float(self.correlation_source['c1'])
        c_2 = float(self.correlation_source['c2'])

        corr = pd.DataFrame(
            np.eye(len(self.locations)),
            index=self.locations,
            columns=self.locations,
            dtype=float,
        )

        for i_loc in self.locations:
            if str(i_loc) not in z_over_h:
                msg = f'Missing z/H value for location `{i_loc}`.'
                raise ValueError(msg)

            x_i = float(z_over_h[str(i_loc)])

            for j_loc in self.locations:
                x_j = float(z_over_h[str(j_loc)])
                corr.loc[i_loc, j_loc] = np.exp(-c_1 * abs(x_i - x_j) ** c_2)

        return corr

    def _estimate_model_corr(self, data: pd.DataFrame) -> pd.DataFrame:
        rid_matrix = data['RID'].unstack(['loc', 'dir']).copy()

        if self.fit_directions_together:
            rid_matrix = rid_matrix.groupby(level='loc', axis=1).mean()
            rid_matrix = rid_matrix.reindex(columns=self.locations)

            if rid_matrix.shape[0] < 2:
                return pd.DataFrame(
                    np.eye(len(self.locations)),
                    index=self.locations,
                    columns=self.locations,
                    dtype=float,
                )

            rank_u = rid_matrix.rank(axis=0, method='average') / (
                len(rid_matrix) + 1.0
            )
            z_vals = pd.DataFrame(
                sp.stats.norm.ppf(rank_u.to_numpy()),
                index=rid_matrix.index,
                columns=rid_matrix.columns,
            )
            return z_vals.corr().reindex(
                index=self.locations, columns=self.locations
            )

        corr_by_dir = []
        for direction in rid_matrix.columns.get_level_values('dir').unique():
            rid_dir = rid_matrix.xs(direction, axis=1, level='dir')
            rid_dir = rid_dir.reindex(columns=self.locations)

            if rid_dir.shape[0] < 2:
                continue

            rank_u = rid_dir.rank(axis=0, method='average') / (len(rid_dir) + 1.0)
            z_vals = pd.DataFrame(
                sp.stats.norm.ppf(rank_u.to_numpy()),
                index=rid_dir.index,
                columns=rid_dir.columns,
            )
            corr_by_dir.append(z_vals.corr())

        if not corr_by_dir:
            return pd.DataFrame(
                np.eye(len(self.locations)),
                index=self.locations,
                columns=self.locations,
                dtype=float,
            )

        avg_corr = sum(corr_by_dir) / len(corr_by_dir)
        return avg_corr.reindex(index=self.locations, columns=self.locations)

    def fit(self, psdr_rsdr_data: pd.DataFrame) -> None:
        """
        Fit the residual drift model to structural analysis data.

        Parameters
        ----------
        psdr_rsdr_data: DataFrame
            Structural analysis data used to fit the RID|PID models.
            The index must include `loc` and `dir` and the columns
            must include `PID` and `RID`.
        """
        if not isinstance(psdr_rsdr_data.index, pd.MultiIndex):
            msg = '`psdr_rsdr_data` must have a MultiIndex index.'
            raise TypeError(msg)

        if (
            'loc' not in psdr_rsdr_data.index.names
            or 'dir' not in psdr_rsdr_data.index.names
        ):
            msg = '`psdr_rsdr_data` index must include `loc` and `dir`.'
            raise ValueError(msg)

        missing_cols = {'PID', 'RID'} - set(psdr_rsdr_data.columns)
        if missing_cols:
            msg = (
                '`psdr_rsdr_data` must include `PID` and `RID` columns. '
                f'Missing: {sorted(missing_cols)}.'
            )
            raise ValueError(msg)

        data = psdr_rsdr_data[['PID', 'RID']].copy().astype(float).abs()

        self.locations = sorted(
            str(loc) for loc in data.index.get_level_values('loc').unique()
        )
        self.directions = sorted(
            str(dr) for dr in data.index.get_level_values('dir').unique()
        )

        self.models_by_loc = {}
        self.models_by_loc_dir = {}
        self.model_maxmax = None
        self.models_maxmax_by_dir = {}
        self.correlation_matrix = None

        if self.approach == 'max_max':
            group_levels = [name for name in data.index.names if name not in {'loc'}]
            if not group_levels:
                msg = (
                    'For `max_max`, `psdr_rsdr_data` must include index levels '
                    'besides `loc` so spatial maxima can be formed.'
                )
                raise ValueError(msg)

            data_maxmax = data.groupby(group_levels).max()

            if self.fit_directions_together:
                self.model_maxmax = self._fit_single_model(
                    raw_pid=data_maxmax['PID'].to_numpy(),
                    raw_rid=data_maxmax['RID'].to_numpy(),
                    censoring_limit=self.censoring_limit,
                    model_fit_method=self.model_fit_method,
                )
            else:
                if 'dir' not in data_maxmax.index.names:
                    msg = (
                        'For `max_max` with `fit_directions_together=False`, '
                        'the grouped spatial maxima must retain a `dir` index level.'
                    )
                    raise ValueError(msg)

                self.models_maxmax_by_dir = {}

                for direction in sorted(
                    str(dr)
                    for dr in data_maxmax.index.get_level_values('dir').unique()
                ):
                    data_maxmax_dir = data_maxmax[
                        data_maxmax.index.get_level_values('dir').astype(str)
                        == direction
                    ]

                    if data_maxmax_dir.empty:
                        msg = (
                            'No spatial-maxima data available for '
                            f'direction `{direction}`.'
                        )
                        raise ValueError(msg)

                    self.models_maxmax_by_dir[direction] = self._fit_single_model(
                        raw_pid=data_maxmax_dir['PID'].to_numpy(),
                        raw_rid=data_maxmax_dir['RID'].to_numpy(),
                        censoring_limit=self.censoring_limit,
                        model_fit_method=self.model_fit_method,
                    )

            self.fitted = True
            return

        if self.fit_directions_together:
            for location in self.locations:
                data_loc = data[
                    data.index.get_level_values('loc').astype(str) == location
                ]

                if data_loc.empty:
                    msg = (
                        'No structural analysis data available for '
                        f'location `{location}`.'
                    )
                    raise ValueError(msg)

                self.models_by_loc[location] = self._fit_single_model(
                    raw_pid=data_loc['PID'].to_numpy(),
                    raw_rid=data_loc['RID'].to_numpy(),
                    censoring_limit=self.censoring_limit,
                    model_fit_method=self.model_fit_method,
                )

        else:
            for location in self.locations:
                for direction in self.directions:
                    mask = (
                        data.index.get_level_values('loc').astype(str) == location
                    ) & (data.index.get_level_values('dir').astype(str) == direction)
                    data_loc_dir = data[mask]

                    if data_loc_dir.empty:
                        msg = (
                            'No structural analysis data available for '
                            f'location `{location}` and direction `{direction}`.'
                        )
                        raise ValueError(msg)

                    self.models_by_loc_dir[location, direction] = (
                        self._fit_single_model(
                            raw_pid=data_loc_dir['PID'].to_numpy(),
                            raw_rid=data_loc_dir['RID'].to_numpy(),
                            censoring_limit=self.censoring_limit,
                            model_fit_method=self.model_fit_method,
                        )
                    )

        if self.correlation_source == 'model':
            self.correlation_matrix = self._estimate_model_corr(data)
        else:
            self.correlation_matrix = self._estimate_reference_corr()

        assert self.correlation_matrix is not None
        self.correlation_matrix = self.correlation_matrix.fillna(0.0)
        self.fitted = True

    def sample(
        self,
        pid: pd.DataFrame,
        rng: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """
        Generate RID realizations for a PID sample.

        Parameters
        ----------
        pid: DataFrame
            PID realizations. Columns must use a MultiIndex with
            levels `edp_type`, `loc`, and `dir` after selecting the
            PID block.
        rng: Generator, optional
            Random number generator used for copula sampling.

        Returns
        -------
        DataFrame
            RID realizations with columns labeled as
            `('RID', loc, dir)`.
        """
        if not self.fitted:
            msg = 'Residual drift model has not been fitted yet.'
            raise ValueError(msg)

        if not isinstance(pid.columns, pd.MultiIndex):
            msg = '`pid` must have MultiIndex columns.'
            raise TypeError(msg)

        if 'loc' not in pid.columns.names or 'dir' not in pid.columns.names:
            msg = '`pid` columns must include `loc` and `dir`.'
            raise ValueError(msg)

        pid_df = pid.copy().astype(float)

        pid_locations = sorted(
            str(loc) for loc in pid_df.columns.get_level_values('loc').unique()
        )
        pid_directions = sorted(
            str(dr) for dr in pid_df.columns.get_level_values('dir').unique()
        )

        if self.approach == 'max_max':
            if len(pid_directions) != 1:
                msg = '`max_max` expects PID samples for exactly one direction at a time.'
                raise ValueError(msg)

            direction = pid_directions[0]
            pid_max = pid_df.max(axis=1).to_numpy()

            if self.fit_directions_together:
                assert self.model_maxmax is not None
                model = self.model_maxmax
            else:
                if direction not in self.models_maxmax_by_dir:
                    msg = (
                        'No fitted max-max model available for '
                        f'direction `{direction}`.'
                    )
                    raise ValueError(msg)
                model = self.models_maxmax_by_dir[direction]

            rid_vals = model.generate_rid_samples(pid_max)

            return pd.DataFrame(
                rid_vals,
                index=pid_df.index,
                columns=pd.MultiIndex.from_tuples(
                    [('RID', '0', direction)],
                    names=['edp_type', 'loc', 'dir'],
                ),
            )

        missing_locations = set(pid_locations) - set(self.locations)
        if missing_locations:
            msg = (
                'PID sample contains locations not present in the fitted model: '
                f'{sorted(missing_locations)}.'
            )
            raise ValueError(msg)

        use_rng = rng if rng is not None else np.random.default_rng()

        assert self.correlation_matrix is not None
        corr_loc = self.correlation_matrix.reindex(
            index=pid_locations, columns=pid_locations
        ).fillna(0.0)
        corr_values = self._nearest_psd(corr_loc.to_numpy())
        l_mat = np.linalg.cholesky(corr_values)

        pid_loc_level = pid_df.columns.names.index('loc')
        pid_dir_level = pid_df.columns.names.index('dir')

        rid_cols: list[pd.Series] = []

        for direction in pid_directions:
            u = use_rng.uniform(0.0, 1.0, size=(len(pid_df), len(pid_locations)))
            z = sp.stats.norm.ppf(u)
            z_corr = (l_mat @ z.T).T
            u_corr = sp.stats.norm.cdf(z_corr)

            for i_loc, location in enumerate(pid_locations):
                mask = [
                    str(col[pid_loc_level]) == location
                    and str(col[pid_dir_level]) == direction
                    for col in pid_df.columns
                ]
                pid_series = pid_df.loc[:, mask]

                if pid_series.shape[1] != 1:
                    msg = (
                        'Expected exactly one PID column for '
                        f'location `{location}` and direction `{direction}`.'
                    )
                    raise ValueError(msg)

                pid_vals = pid_series.iloc[:, 0].to_numpy()

                if self.fit_directions_together:
                    model = self.models_by_loc[location]
                else:
                    model = self.models_by_loc_dir[location, direction]

                model.uniform_sample = u_corr[:, i_loc]
                rid_vals = model.generate_rid_samples(pid_vals)

                rid_cols.append(
                    pd.Series(
                        rid_vals,
                        index=pid_df.index,
                        name=('RID', location, direction),
                    )
                )

        rid = pd.concat(rid_cols, axis=1)
        rid.columns = pd.MultiIndex.from_tuples(
            rid.columns,
            names=['edp_type', 'loc', 'dir'],
        )
        return rid

    def diagnostic_plots(
        self,
        figsize: tuple[float, float] = (4.5, 3.0),
        max_cols: int = 3,
        rolling: bool = True,
        training: bool = True,
        model: bool = True,
        show_parameters: bool = True,
        parameter_format: str = '.4g',
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Plot fitted sub-models for visual inspection.

        Returns
        -------
        tuple
            Matplotlib figure and axes array.
        """
        if not self.fitted:
            msg = 'Residual drift model has not been fitted yet.'
            raise ValueError(msg)

        entries: list[tuple[str, Model_Trilinear_Weibull]] = []

        if self.approach == 'max_max':
            if self.fit_directions_together:
                assert self.model_maxmax is not None
                entries.append(('max_max', self.model_maxmax))
            else:
                for direction in self.directions:
                    model_i = self.models_maxmax_by_dir.get(direction)
                    if model_i is not None:
                        entries.append((f'max_max | dir={direction}', model_i))
        else:
            if self.fit_directions_together:
                for location in self.locations:
                    model_i = self.models_by_loc.get(location)
                    if model_i is not None:
                        entries.append((f'loc={location}', model_i))
            else:
                for location in self.locations:
                    for direction in self.directions:
                        model_i = self.models_by_loc_dir.get((location, direction))
                        if model_i is not None:
                            entries.append(
                                (f'loc={location} | dir={direction}', model_i)
                            )

        if not entries:
            msg = 'No fitted sub-models available for plotting.'
            raise ValueError(msg)

        n_plots = len(entries)
        n_cols = min(max_cols, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))

        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
            squeeze=False,
        )

        for ax, (title, model_i) in zip(axs.flat, entries):
            model_i.plot_model(
                ax=ax,
                rolling=rolling,
                training=training,
                model=model,
            )

            if show_parameters:
                parameter_lines = []
                if (
                    getattr(model_i, 'parameter_names', None) is not None
                    and getattr(model_i, 'parameters', None) is not None
                ):
                    parameter_lines = [
                        f'{name}={value:{parameter_format}}'
                        for name, value in zip(
                            model_i.parameter_names,
                            model_i.parameters,
                            strict=False,
                        )
                    ]
                elif getattr(model_i, 'parameters', None) is not None:
                    parameter_lines = [
                        f'p{i}={value:{parameter_format}}'
                        for i, value in enumerate(model_i.parameters)
                    ]

                if parameter_lines:
                    title = title + '\n' + '\n'.join(parameter_lines)

            ax.set_title(title)
            ax.set_xlabel('PID')
            ax.set_ylabel('RID')

            if (
                getattr(model_i, 'raw_pid', None) is not None
                and getattr(model_i, 'raw_rid', None) is not None
            ):
                max_pid = float(np.max(model_i.raw_pid))
                max_rid = float(np.max(model_i.raw_rid))

                ax.set_xlim(0.0, 1.05 * max_pid)
                ax.set_ylim(0.0, 1.05 * max_rid)

        for ax in axs.flat[n_plots:]:
            ax.set_visible(False)

        fig.tight_layout()
        return fig, axs
