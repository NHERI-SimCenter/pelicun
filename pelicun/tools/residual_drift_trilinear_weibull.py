#
# Copyright (c) 2025 Leland Stanford Junior University
# Copyright (c) 2025 The Regents of the University of California
#
# This file is part of pelicun.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# pelicun. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay
# Ioannis Vouvakis Manousakis

"""Residual drift models conditioned on peak transient drift."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
from scipy.optimize import OptimizeResult, basinhopping, minimize

if TYPE_CHECKING:
    from matplotlib.axes import Axes


PDF_FLOOR = 1.0e-6
TINY_FLOAT = 1.0e-300
SMALL_SERIES_LIMIT = 1.0e-6
MAX_EXP_LOG = 700.0
MIN_EXP_LOG = -745.0
MIN_EIGENVALUE = 1.0e-8
MIN_CORRELATION_OBSERVATIONS = 2
DEFAULT_MODEL_PID_MAX = 0.08
DEFAULT_MODEL_PID_POINTS = 1000
DEFAULT_LOCAL_OPTIMIZATION_REPEATS = 3
DEFAULT_BASINHOPPING_ITERATIONS = 20
DEFAULT_BASINHOPPING_TEMPERATURE = 1.0
DEFAULT_BASINHOPPING_STEPSIZE = 0.02
DEFAULT_BASINHOPPING_SUCCESS_ITERATIONS = 5
DEFAULT_GLOBAL_OPTIMIZER_MAX_ITER = 200
DEFAULT_LOCAL_OPTIMIZER_MAX_ITER = 10000
DEFAULT_GLOBAL_OPTIMIZER_TOL = 1.0e-6
DEFAULT_LOCAL_OPTIMIZER_TOL = 1.0e-10
EXPECTED_PID_COLUMN_COUNT = 1
AXIS_PADDING_FACTOR = 1.05

OptimizationMethod = Literal['mle', 'quantiles', 'mle-fast']
ObjectiveFunction = Callable[[npt.NDArray], float | tuple[float, npt.NDArray]]


class Model(ABC):
    """Base class for conditional residual drift models."""

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
        self.fit_meta: OptimizeResult | None = None

        self.rolling_pid: npt.NDArray | None = None
        self.rolling_rid_50: npt.NDArray | None = None
        self.rolling_rid_20: npt.NDArray | None = None
        self.rolling_rid_80: npt.NDArray | None = None

    def add_data(self, raw_pid: npt.NDArray, raw_rid: npt.NDArray) -> None:
        """Store the raw PID and RID samples used for fitting."""
        self.raw_pid = raw_pid
        self.raw_rid = raw_rid

    def calculate_rolling_quantiles(
        self,
        fraction: float = 0.075,
        min_points: int = 5,
    ) -> None:
        """Calculate empirical rolling RID quantiles sorted by PID."""
        assert self.raw_pid is not None
        assert self.raw_rid is not None

        idsort = np.argsort(self.raw_pid)
        pid_vals_sorted = self.raw_pid[idsort]
        rid_vals_sorted = self.raw_rid[idsort]

        num_vals = len(pid_vals_sorted)
        group_size = max(int(num_vals * fraction), min_points)
        window_start_ids = range(num_vals - group_size + 1)

        self.rolling_pid = np.array(
            [np.mean(pid_vals_sorted[i : i + group_size]) for i in window_start_ids]
        )
        self.rolling_rid_50 = np.array(
            [
                np.quantile(rid_vals_sorted[i : i + group_size], 0.50)
                for i in window_start_ids
            ]
        )
        self.rolling_rid_20 = np.array(
            [
                np.quantile(rid_vals_sorted[i : i + group_size], 0.20)
                for i in window_start_ids
            ]
        )
        self.rolling_rid_80 = np.array(
            [
                np.quantile(rid_vals_sorted[i : i + group_size], 0.80)
                for i in window_start_ids
            ]
        )

    def fit(
        self,
        method: OptimizationMethod = 'mle',
        *,
        global_search: bool = True,
    ) -> None:
        """Fit the model parameters.

        Parameters
        ----------
        method
            Optimization objective to use.
        global_search
            If true, use basin hopping before the local optimizer.

        Raises
        ------
        RuntimeError
            If the optimizer fails to converge.
        """
        use_objective, jac, optimizer_method = self._optimizer_settings(method)

        if global_search:
            result = basinhopping(
                func=use_objective,
                x0=self.parameters,
                niter=DEFAULT_BASINHOPPING_ITERATIONS,
                T=DEFAULT_BASINHOPPING_TEMPERATURE,
                stepsize=DEFAULT_BASINHOPPING_STEPSIZE,
                niter_success=DEFAULT_BASINHOPPING_SUCCESS_ITERATIONS,
                minimizer_kwargs={
                    'method': optimizer_method,
                    'jac': jac,
                    'bounds': self.parameter_bounds,
                    'options': {'maxiter': DEFAULT_GLOBAL_OPTIMIZER_MAX_ITER},
                    'tol': DEFAULT_GLOBAL_OPTIMIZER_TOL,
                },
            )
        else:
            result = self._run_local_optimization(
                use_objective,
                jac=jac,
                optimizer_method=optimizer_method,
            )

        self.fit_meta = result
        if not result.success:
            msg = 'Minimization failed.'
            raise RuntimeError(msg)
        self.parameters = result.x
        self.fit_status = True

    def get_mle_objective(self, parameters: npt.NDArray) -> float:
        """Evaluate the negative log-likelihood objective.

        Returns
        -------
        float
            Negative log-likelihood value.
        """
        self.parameters = parameters
        density = self.evaluate_pdf(self.raw_rid, self.raw_pid, self.censoring_limit)
        return -float(np.sum(np.log(density)))

    def get_quantile_objective(self, parameters: npt.NDArray) -> float:
        """Evaluate the empirical-quantile least-squares objective.

        Returns
        -------
        float
            Least-squares loss between empirical and model quantiles.
        """
        self.parameters = parameters

        if self.rolling_pid is None:
            self.calculate_rolling_quantiles()

        assert self.rolling_pid is not None
        assert self.rolling_rid_50 is not None
        assert self.rolling_rid_20 is not None
        assert self.rolling_rid_80 is not None

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
        return float(loss)

    def generate_rid_samples(self, pid_samples: npt.NDArray) -> npt.NDArray:
        """Generate RID samples from PID samples using inverse transform sampling.

        Returns
        -------
        npt.NDArray
            Simulated RID samples.
        """
        if self.uniform_sample is None:
            self.uniform_sample = np.random.uniform(0.00, 1.00, len(pid_samples))

        rid_samples = self.evaluate_inverse_cdf(self.uniform_sample, pid_samples)
        self.sim_pid = pid_samples
        self.sim_rid = rid_samples
        return rid_samples

    def plot_data(
        self,
        ax: Axes | None = None,
        scatter_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Add the raw data scatter plot to an axis or show a new figure."""
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
            return

        ax.scatter(self.raw_pid, self.raw_rid, **scatter_kwargs)

    def plot_model(
        self,
        ax: Axes,
        *,
        rolling: bool = True,
        training: bool = True,
        model: bool = True,
        model_color: str = 'C0',
    ) -> None:
        """Plot training data, empirical quantiles, and fitted model quantiles."""
        if not self.fit_status:
            self.fit()

        if training:
            self.plot_data(ax)

        if rolling:
            self.calculate_rolling_quantiles()
            ax.plot(self.rolling_pid, self.rolling_rid_50, 'k')
            ax.plot(self.rolling_pid, self.rolling_rid_20, 'k', linestyle='dashed')
            ax.plot(self.rolling_pid, self.rolling_rid_80, 'k', linestyle='dashed')

        if model:
            model_pid = np.linspace(
                0.00,
                DEFAULT_MODEL_PID_MAX,
                DEFAULT_MODEL_PID_POINTS,
            )
            model_rid_50 = self.evaluate_inverse_cdf(0.50, model_pid)
            model_rid_20 = self.evaluate_inverse_cdf(0.20, model_pid)
            model_rid_80 = self.evaluate_inverse_cdf(0.80, model_pid)

            ax.plot(model_pid, model_rid_50, model_color)
            ax.plot(model_pid, model_rid_20, model_color, linestyle='dashed')
            ax.plot(model_pid, model_rid_80, model_color, linestyle='dashed')

    @abstractmethod
    def evaluate_inverse_cdf(
        self,
        _quantile: npt.NDArray | float,
        _pid: npt.NDArray,
    ) -> npt.NDArray:
        """Evaluate the inverse conditional RID|PID CDF.

        Returns
        -------
        npt.NDArray
            Conditional quantile values.

        Raises
        ------
        NotImplementedError
            Always, unless a subclass overrides this method.
        """
        raise self._subclass_implementation_error()

    @abstractmethod
    def evaluate_pdf(
        self,
        _rid: npt.NDArray,
        _pid: npt.NDArray,
        _censoring_limit: float | None = None,
    ) -> npt.NDArray:
        """Evaluate the conditional RID|PID PDF.

        Returns
        -------
        npt.NDArray
            Conditional density values.

        Raises
        ------
        NotImplementedError
            Always, unless a subclass overrides this method.
        """
        raise self._subclass_implementation_error()

    @abstractmethod
    def evaluate_cdf(self, _rid: npt.NDArray, _pid: npt.NDArray) -> npt.NDArray:
        """Evaluate the conditional RID|PID CDF.

        Returns
        -------
        npt.NDArray
            Conditional cumulative probability values.

        Raises
        ------
        NotImplementedError
            Always, unless a subclass overrides this method.
        """
        raise self._subclass_implementation_error()

    @staticmethod
    def trilinear_fnc(
        pid: npt.NDArray,
        y0: float,
        m0: float,
        m1: float,
        m2: float,
        x0: float,
        x1: float,
    ) -> npt.NDArray:
        """Evaluate a continuous trilinear function.

        Returns
        -------
        npt.NDArray
            Trilinear function values evaluated at `pid`.
        """
        y1 = y0 + m0 * x0
        y2 = y1 + m1 * (x1 - x0)
        res = m0 * pid + y0

        mask = pid > x0
        res[mask] = (pid[mask] - x0) * m1 + y1

        mask = pid > x1
        res[mask] = (pid[mask] - x1) * m2 + y2
        return res

    @abstractmethod
    def get_mle_objective_fast(
        self,
        _parameters: npt.NDArray,
    ) -> tuple[float, npt.NDArray]:
        """Evaluate the fast MLE objective and gradient.

        Returns
        -------
        tuple[float, npt.NDArray]
            Objective value and gradient vector.

        Raises
        ------
        NotImplementedError
            Always, unless a subclass overrides this method.
        """
        raise self._subclass_implementation_error()

    def _subclass_implementation_error(self) -> NotImplementedError:
        return NotImplementedError(
            f'{type(self).__name__} must implement this method.'
        )

    def _optimizer_settings(
        self,
        method: OptimizationMethod,
    ) -> tuple[ObjectiveFunction, bool | None, str]:
        if method == 'quantiles':
            return self.get_quantile_objective, None, 'Nelder-Mead'
        if method == 'mle':
            return self.get_mle_objective, None, 'Nelder-Mead'
        if method == 'mle-fast':
            return self.get_mle_objective_fast, True, 'L-BFGS-B'

        msg = f'Invalid method: `{method}`.'
        raise ValueError(msg)

    def _run_local_optimization(
        self,
        use_objective: ObjectiveFunction,
        *,
        jac: bool | None,
        optimizer_method: str,
    ) -> OptimizeResult:
        repeats_remaining = DEFAULT_LOCAL_OPTIMIZATION_REPEATS
        result: OptimizeResult | None = None
        success = False

        while not success and repeats_remaining > 0:
            result = minimize(
                fun=use_objective,
                x0=self.parameters,
                jac=jac,
                bounds=self.parameter_bounds,
                method=optimizer_method,
                options={'maxiter': DEFAULT_LOCAL_OPTIMIZER_MAX_ITER},
                tol=DEFAULT_LOCAL_OPTIMIZER_TOL,
            )
            success = result.success
            repeats_remaining -= 1

        assert result is not None
        return result


class Model_Trilinear_Weibull(Model):
    """Conditional Weibull model with trilinear parameter functions."""

    def __init__(self) -> None:
        super().__init__()
        self.parameters = np.array((0.005, 0.15, 0.020 - 0.005, 0.25, 1.20, 1.20))
        self.parameter_names = [
            'pid_0',
            'lambda_slope_1',
            'pid_1_minus_pid_0',
            'lambda_slope_2_minus_lambda_slope_1',
            'kappa_1',
            'kappa_2',
        ]
        self.parameter_bounds = [
            (1e-4, 0.05),
            (0.00, 1.00),
            (1e-3, 0.1),
            (0.00, 1.00),
            (1.0, 10.0),
            (1.0, 10.0),
        ]

    def obtain_lambda(self, pid: npt.NDArray) -> npt.NDArray:
        """Evaluate the Weibull scale parameter for each PID value.

        Returns
        -------
        npt.NDArray
            Weibull scale parameter values.
        """
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
        lambda_0 = PDF_FLOOR
        lambda_1 = lambda_0 + lambda_slope_1 * pid_1_minus_pid_0

        lambda_trilinear = np.empty_like(pid)
        mask = pid <= pid_0
        lambda_trilinear[mask] = PDF_FLOOR

        mask = np.logical_and(pid_0 < pid, pid <= pid_1)
        lambda_trilinear[mask] = (pid[mask] - pid_0) * lambda_slope_1 + lambda_0

        mask = pid_1 < pid
        lambda_trilinear[mask] = (pid[mask] - pid_1) * lambda_slope_2 + lambda_1
        return lambda_trilinear

    def obtain_kappa(self, pid: npt.NDArray) -> npt.NDArray:
        """Evaluate the Weibull shape parameter for each PID value.

        Returns
        -------
        npt.NDArray
            Weibull shape parameter values.
        """
        assert self.parameters is not None
        (
            pid_0,
            _lambda_slope_1,
            pid_1_minus_pid_0,
            _lambda_slope_2_minus_lambda_slope_1,
            kappa_1,
            kappa_2,
        ) = self.parameters
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

    def obtain_lambda_grad(self, pid: npt.NDArray) -> npt.NDArray:
        """Evaluate gradients of the Weibull scale parameter.

        Returns
        -------
        npt.NDArray
            Gradient matrix of the scale parameter.
        """
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
        lambda_trilinear = np.empty((len(pid), len(self.parameters)))

        mask = pid <= pid_0
        lambda_trilinear[mask, :] = 0.00

        mask = np.logical_and(pid_0 < pid, pid <= pid_0 + pid_1_minus_pid_0)
        lambda_trilinear[mask, 0] = -lambda_slope_1
        lambda_trilinear[mask, 1] = pid[mask] - pid_0
        lambda_trilinear[mask, 2:] = 0.00

        mask = pid_0 + pid_1_minus_pid_0 < pid
        lambda_trilinear[mask, 0] = -lambda_slope_2
        lambda_trilinear[mask, 1] = pid[mask] - pid_0
        lambda_trilinear[mask, 2] = lambda_slope_1 - lambda_slope_2
        lambda_trilinear[mask, 3] = pid[mask] - (pid_0 + pid_1_minus_pid_0)
        lambda_trilinear[mask, 4:] = 0.00

        return lambda_trilinear

    def obtain_kappa_grad(self, pid: npt.NDArray) -> npt.NDArray:
        """Evaluate gradients of the Weibull shape parameter.

        Returns
        -------
        npt.NDArray
            Gradient matrix of the shape parameter.
        """
        assert self.parameters is not None
        (
            pid_0,
            _lambda_slope_1,
            pid_1_minus_pid_0,
            _lambda_slope_2_minus_lambda_slope_1,
            kappa_1,
            kappa_2,
        ) = self.parameters
        kappa_trilinear = np.empty((len(pid), len(self.parameters)))

        mask = pid <= pid_0
        kappa_trilinear[mask, 0:4] = 0.00
        kappa_trilinear[mask, 4] = 1.00
        kappa_trilinear[mask, 5] = 0.00

        mask = np.logical_and(pid_0 < pid, pid <= pid_0 + pid_1_minus_pid_0)
        kappa_trilinear[mask, 0] = -(kappa_2 - kappa_1) / pid_1_minus_pid_0
        kappa_trilinear[mask, 1] = 0.00
        kappa_trilinear[mask, 2] = (
            -(pid[mask] - pid_0) * (kappa_2 - kappa_1) / pid_1_minus_pid_0**2
        )
        kappa_trilinear[mask, 3] = 0.00
        kappa_trilinear[mask, 4] = 1.00 - ((pid[mask] - pid_0) / pid_1_minus_pid_0)
        kappa_trilinear[mask, 5] = (pid[mask] - pid_0) / pid_1_minus_pid_0

        mask = pid_0 + pid_1_minus_pid_0 < pid
        kappa_trilinear[mask, 0:5] = 0.00
        kappa_trilinear[mask, 5] = 1.00

        return kappa_trilinear

    def get_mle_objective_fast(
        self,
        parameters: npt.NDArray,
    ) -> tuple[float, npt.NDArray]:
        """Compute the negative log-likelihood and its gradient.

        Parameters
        ----------
        parameters
            Model parameters.

        Returns
        -------
        tuple[float, npt.NDArray]
            Objective value and gradient vector.

        Raises
        ------
        ValueError
            If the censoring limit is missing or nonpositive.
        """
        self.parameters = parameters

        if self.censoring_limit is None:
            msg = 'censoring_limit must be defined for the stable censored formulation.'
            raise ValueError(msg)

        c_lim = float(self.censoring_limit)
        if c_lim <= 0.0:
            msg = 'censoring_limit must be positive for the stable censored formulation.'
            raise ValueError(msg)

        assert self.raw_pid is not None
        assert self.raw_rid is not None

        mask_censored = self.raw_rid < c_lim
        mask_uncensored = ~mask_censored

        lambda_vec = self.obtain_lambda(self.raw_pid)[:, np.newaxis]
        kappa_vec = self.obtain_kappa(self.raw_pid)[:, np.newaxis]
        rid_vec = self.raw_rid[:, np.newaxis]

        lambda_grad = self.obtain_lambda_grad(self.raw_pid)
        kappa_grad = self.obtain_kappa_grad(self.raw_pid)

        lambda_safe = np.maximum(lambda_vec, TINY_FLOAT)
        kappa_safe = np.maximum(kappa_vec, TINY_FLOAT)

        n_obs = len(self.raw_rid)
        n_params = len(parameters)
        grad_log_l = np.zeros((n_obs, n_params), dtype=float)
        log_l = np.zeros((n_obs, 1), dtype=float)

        self._fill_uncensored_likelihood(
            mask_uncensored,
            rid_vec,
            lambda_safe,
            kappa_safe,
            lambda_grad,
            kappa_grad,
            log_l,
            grad_log_l,
        )
        self._fill_censored_likelihood(
            mask_censored,
            c_lim,
            lambda_safe,
            kappa_safe,
            lambda_grad,
            kappa_grad,
            log_l,
            grad_log_l,
        )
        self._apply_likelihood_floor(log_l, grad_log_l)

        objective = -float(np.sum(log_l))
        gradient = -np.sum(grad_log_l, axis=0)
        return objective, gradient

    def evaluate_pdf(
        self,
        rid: npt.NDArray,
        pid: npt.NDArray,
        censoring_limit: float | None = None,
    ) -> npt.NDArray:
        """Evaluate the conditional Weibull RID|PID PDF.

        Returns
        -------
        npt.NDArray
            Conditional density values.
        """
        lambda_trilinear = self.obtain_lambda(pid)
        kappa_trilinear = self.obtain_kappa(pid)
        pdf_val = sp.stats.weibull_min.pdf(
            rid,
            kappa_trilinear,
            0.00,
            lambda_trilinear,
        )
        if censoring_limit:
            censored_range_mass = self.evaluate_cdf(
                np.full(len(pid), censoring_limit),
                pid,
            )
            mask = rid <= censoring_limit
            pdf_val[mask] = censored_range_mass[mask]
        pdf_val[pdf_val < PDF_FLOOR] = PDF_FLOOR
        return pdf_val

    def evaluate_cdf(self, rid: npt.NDArray, pid: npt.NDArray) -> npt.NDArray:
        """Evaluate the conditional Weibull RID|PID CDF.

        Returns
        -------
        npt.NDArray
            Conditional cumulative probability values.
        """
        lambda_trilinear = self.obtain_lambda(pid)
        kappa_trilinear = self.obtain_kappa(pid)
        return sp.stats.weibull_min.cdf(rid, kappa_trilinear, 0.00, lambda_trilinear)

    def evaluate_inverse_cdf(
        self,
        quantile: npt.NDArray | float,
        pid: npt.NDArray,
    ) -> npt.NDArray:
        """Evaluate the inverse conditional Weibull RID|PID CDF.

        Returns
        -------
        npt.NDArray
            Conditional quantile values.
        """
        lambda_trilinear = self.obtain_lambda(pid)
        kappa_trilinear = self.obtain_kappa(pid)
        return sp.stats.weibull_min.ppf(
            quantile,
            kappa_trilinear,
            0.00,
            lambda_trilinear,
        )

    @staticmethod
    def _apply_likelihood_floor(
        log_l: npt.NDArray,
        grad_log_l: npt.NDArray,
    ) -> None:
        log_l_floor = np.log(PDF_FLOOR)
        clipped_mask = log_l[:, 0] < log_l_floor
        if np.any(clipped_mask):
            log_l[clipped_mask, :] = log_l_floor
            grad_log_l[clipped_mask, :] = 0.0

    @staticmethod
    def _fill_uncensored_likelihood(
        mask_uncensored: npt.NDArray,
        rid_vec: npt.NDArray,
        lambda_safe: npt.NDArray,
        kappa_safe: npt.NDArray,
        lambda_grad: npt.NDArray,
        kappa_grad: npt.NDArray,
        log_l: npt.NDArray,
        grad_log_l: npt.NDArray,
    ) -> None:
        if not np.any(mask_uncensored):
            return

        rid_u = rid_vec[mask_uncensored, :]
        lam_u = lambda_safe[mask_uncensored, :]
        kap_u = kappa_safe[mask_uncensored, :]
        lam_grad_u = lambda_grad[mask_uncensored, :]
        kap_grad_u = kappa_grad[mask_uncensored, :]

        rid_u_safe = np.maximum(rid_u, TINY_FLOAT)
        log_frac_u = np.log(rid_u_safe) - np.log(lam_u)
        log_t_u = kap_u * log_frac_u
        t_u = np.exp(np.clip(log_t_u, MIN_EXP_LOG, MAX_EXP_LOG))

        log_l[mask_uncensored, :] = (
            np.log(kap_u) - np.log(lam_u) + (kap_u - 1.0) * log_frac_u - t_u
        )
        grad_log_l[mask_uncensored, :] = (
            kap_grad_u / kap_u
            + kap_grad_u * log_frac_u * (1.0 - t_u)
            + lam_grad_u / lam_u * kap_u * (t_u - 1.0)
        )

    @staticmethod
    def _fill_censored_likelihood(
        mask_censored: npt.NDArray,
        c_lim: float,
        lambda_safe: npt.NDArray,
        kappa_safe: npt.NDArray,
        lambda_grad: npt.NDArray,
        kappa_grad: npt.NDArray,
        log_l: npt.NDArray,
        grad_log_l: npt.NDArray,
    ) -> None:
        if not np.any(mask_censored):
            return

        lam_c = lambda_safe[mask_censored, :]
        kap_c = kappa_safe[mask_censored, :]
        lam_grad_c = lambda_grad[mask_censored, :]
        kap_grad_c = kappa_grad[mask_censored, :]

        log_c_over_l = np.log(c_lim) - np.log(lam_c)
        log_s_c = kap_c * log_c_over_l
        s_c = np.exp(np.clip(log_s_c, MIN_EXP_LOG, MAX_EXP_LOG))

        log_l[mask_censored, :] = np.log(-np.expm1(-s_c))
        omega_c = Model_Trilinear_Weibull._stable_omega(s_c)
        grad_log_l[mask_censored, :] = omega_c * (
            log_c_over_l * kap_grad_c - (kap_c / lam_c) * lam_grad_c
        )

    @staticmethod
    def _stable_omega(s_c: npt.NDArray) -> npt.NDArray:
        omega_c = np.empty_like(s_c)
        small_mask = s_c < SMALL_SERIES_LIMIT
        large_mask = s_c > MAX_EXP_LOG
        mid_mask = ~small_mask & ~large_mask

        if np.any(small_mask):
            s_small = s_c[small_mask]
            omega_c[small_mask] = 1.0 - 0.5 * s_small + (s_small * s_small) / 12.0

        if np.any(mid_mask):
            s_mid = s_c[mid_mask]
            omega_c[mid_mask] = s_mid / np.expm1(s_mid)

        if np.any(large_mask):
            omega_c[large_mask] = 0.0

        return omega_c


class ResidualDriftTrilinearWeibullModel:
    """Building-level residual drift model."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config.copy()

        self.approach = self.config['approach']
        self.fit_directions_together = self.config.get(
            'fit_directions_together',
            False,
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

    def fit(self, psdr_rsdr_data: pd.DataFrame) -> None:
        """Fit the residual drift model to structural analysis data.

        Parameters
        ----------
        psdr_rsdr_data
            Structural analysis data used to fit the RID|PID models. The index
            must include `loc` and `dir`, and the columns must include `PID`
            and `RID`.

        """
        self._validate_fit_input(psdr_rsdr_data)
        data = psdr_rsdr_data[['PID', 'RID']].copy().astype(float).abs()
        self._initialize_fit_state(data)

        if self.approach == 'max_max':
            self._fit_max_max_models(data)
            self.fitted = True
            return

        self._fit_story_models(data)
        self.correlation_matrix = self._estimate_corr(data)
        self.fitted = True

    def sample(
        self,
        pid: pd.DataFrame,
        rng: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Generate RID realizations for a PID sample.

        Parameters
        ----------
        pid
            PID realizations. Columns must use a MultiIndex with levels `loc`
            and `dir`.
        rng
            Random number generator used for copula sampling.

        Returns
        -------
        DataFrame
            RID realizations with columns labeled as `('RID', loc, dir)`.

        """
        self._validate_sample_input(pid)
        pid_df = pid.copy().astype(float)
        pid_locations = self._sorted_column_values(pid_df, 'loc')
        pid_directions = self._sorted_column_values(pid_df, 'dir')

        if self.approach == 'max_max':
            return self._sample_max_max(pid_df, pid_directions)

        self._check_sample_locations(pid_locations)
        use_rng = rng if rng is not None else np.random.default_rng()
        return self._sample_story(pid_df, pid_locations, pid_directions, use_rng)

    def diagnostic_plots(
        self,
        figsize: tuple[float, float] = (4.5, 3.0),
        max_cols: int = 3,
        *,
        rolling: bool = True,
        training: bool = True,
        model: bool = True,
        show_parameters: bool = True,
        parameter_format: str = '.4g',
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot fitted sub-models for visual inspection.

        Parameters
        ----------
        figsize
            Size of each subplot cell.
        max_cols
            Maximum number of subplot columns.
        rolling
            If true, show empirical rolling quantiles.
        training
            If true, show the training data.
        model
            If true, show fitted model quantiles.
        show_parameters
            If true, include fitted parameters in subplot titles.
        parameter_format
            Format specification for parameter values.

        Returns
        -------
        tuple[Figure, ndarray]
            Matplotlib figure and axes array.

        Raises
        ------
        ValueError
            If the model is not fitted or has no sub-models available.
        """
        if not self.fitted:
            msg = 'Residual drift model has not been fitted yet.'
            raise ValueError(msg)

        entries = self._diagnostic_entries()
        if not entries:
            msg = 'No fitted sub-models available for plotting.'
            raise ValueError(msg)

        fig, axs = self._create_diagnostic_axes(len(entries), figsize, max_cols)
        for ax, (plot_title, model_i) in zip(axs.flat, entries):
            self._plot_diagnostic_entry(
                ax,
                plot_title,
                model_i,
                rolling=rolling,
                training=training,
                model=model,
                show_parameters=show_parameters,
                parameter_format=parameter_format,
            )

        for ax in axs.flat[len(entries) :]:
            ax.set_visible(False)

        fig.tight_layout()
        return fig, axs

    def _validate_config(self) -> None:
        if self.approach not in {'story', 'max_max'}:
            msg = f'Invalid approach: `{self.approach}`.'
            raise ValueError(msg)

        if self.correlation_source == 'model':
            return

        if not isinstance(self.correlation_source, dict):
            msg = (
                '`correlation_source` must be `model` or a dictionary defining '
                'the reference correlation model.'
            )
            raise TypeError(msg)

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
            raise TypeError(msg)

    @staticmethod
    def _nearest_psd(mat: np.ndarray) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(mat)
        eigvals[eigvals < MIN_EIGENVALUE] = MIN_EIGENVALUE
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
        model_fit_method: OptimizationMethod,
        *,
        global_search: bool = True,
    ) -> Model_Trilinear_Weibull:
        model = Model_Trilinear_Weibull()
        model.add_data(raw_pid=raw_pid, raw_rid=raw_rid)
        model.censoring_limit = censoring_limit
        model.fit(method=model_fit_method, global_search=global_search)
        return model

    @staticmethod
    def _validate_fit_input(psdr_rsdr_data: pd.DataFrame) -> None:
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

    def _initialize_fit_state(self, data: pd.DataFrame) -> None:
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
        self.fitted = False

    def _fit_max_max_models(self, data: pd.DataFrame) -> None:
        group_levels = [name for name in data.index.names if name != 'loc']
        if not group_levels:
            msg = (
                'For `max_max`, `psdr_rsdr_data` must include index levels besides '
                '`loc` so spatial maxima can be formed.'
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
            return

        self._fit_max_max_models_by_direction(data_maxmax)

    def _fit_max_max_models_by_direction(self, data_maxmax: pd.DataFrame) -> None:
        if 'dir' not in data_maxmax.index.names:
            msg = (
                'For `max_max` with `fit_directions_together=False`, the grouped '
                'spatial maxima must retain a `dir` index level.'
            )
            raise ValueError(msg)

        self.models_maxmax_by_dir = {}
        for direction in self._sorted_index_values(data_maxmax, 'dir'):
            data_maxmax_dir = self._data_for_index_value(
                data_maxmax, 'dir', direction
            )
            if data_maxmax_dir.empty:
                msg = (
                    f'No spatial-maxima data available for direction `{direction}`.'
                )
                raise ValueError(msg)

            self.models_maxmax_by_dir[direction] = self._fit_single_model(
                raw_pid=data_maxmax_dir['PID'].to_numpy(),
                raw_rid=data_maxmax_dir['RID'].to_numpy(),
                censoring_limit=self.censoring_limit,
                model_fit_method=self.model_fit_method,
            )

    def _fit_story_models(self, data: pd.DataFrame) -> None:
        if self.fit_directions_together:
            self._fit_story_models_by_location(data)
            return

        self._fit_story_models_by_location_and_direction(data)

    def _fit_story_models_by_location(self, data: pd.DataFrame) -> None:
        for location in self.locations:
            data_loc = self._data_for_index_value(data, 'loc', location)
            if data_loc.empty:
                msg = f'No structural analysis data available for location `{location}`.'
                raise ValueError(msg)

            self.models_by_loc[location] = self._fit_single_model(
                raw_pid=data_loc['PID'].to_numpy(),
                raw_rid=data_loc['RID'].to_numpy(),
                censoring_limit=self.censoring_limit,
                model_fit_method=self.model_fit_method,
            )

    def _fit_story_models_by_location_and_direction(
        self, data: pd.DataFrame
    ) -> None:
        for location in self.locations:
            for direction in self.directions:
                data_loc_dir = self._data_for_loc_dir(data, location, direction)
                if data_loc_dir.empty:
                    msg = (
                        'No structural analysis data available for '
                        f'location `{location}` and direction `{direction}`.'
                    )
                    raise ValueError(msg)

                self.models_by_loc_dir[location, direction] = self._fit_single_model(
                    raw_pid=data_loc_dir['PID'].to_numpy(),
                    raw_rid=data_loc_dir['RID'].to_numpy(),
                    censoring_limit=self.censoring_limit,
                    model_fit_method=self.model_fit_method,
                )

    def _estimate_corr(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.correlation_source == 'model':
            corr = self._estimate_model_corr(data)
        else:
            corr = self._estimate_reference_corr()
        return corr.fillna(0.0)

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
        rid_matrix = self._rid_matrix(data)

        if self.fit_directions_together:
            return self._estimate_location_corr(rid_matrix)

        corr_by_dir = []
        for direction in rid_matrix.columns.get_level_values('dir').unique():
            rid_dir = rid_matrix.xs(direction, axis=1, level='dir')
            rid_dir = rid_dir.reindex(columns=self.locations)

            if rid_dir.shape[0] < MIN_CORRELATION_OBSERVATIONS:
                continue

            corr_by_dir.append(self._normal_score_corr(rid_dir))

        if not corr_by_dir:
            return self._identity_corr()

        avg_corr = sum(corr_by_dir) / len(corr_by_dir)
        return avg_corr.reindex(index=self.locations, columns=self.locations)

    def _estimate_location_corr(self, rid_matrix: pd.DataFrame) -> pd.DataFrame:
        rid_by_loc = rid_matrix.groupby(level='loc', axis=1).mean()
        rid_by_loc = rid_by_loc.reindex(columns=self.locations)

        if rid_by_loc.shape[0] < MIN_CORRELATION_OBSERVATIONS:
            return self._identity_corr()

        return self._normal_score_corr(rid_by_loc).reindex(
            index=self.locations,
            columns=self.locations,
        )

    def _identity_corr(self) -> pd.DataFrame:
        return pd.DataFrame(
            np.eye(len(self.locations)),
            index=self.locations,
            columns=self.locations,
            dtype=float,
        )

    @staticmethod
    def _normal_score_corr(values: pd.DataFrame) -> pd.DataFrame:
        rank_u = values.rank(axis=0, method='average') / (len(values) + 1.0)
        z_vals = pd.DataFrame(
            sp.stats.norm.ppf(rank_u.to_numpy()),
            index=values.index,
            columns=values.columns,
        )
        return z_vals.corr()

    @staticmethod
    def _rid_matrix(data: pd.DataFrame) -> pd.DataFrame:
        index_names = [
            name for name in data.index.names if name not in {'loc', 'dir'}
        ]
        if not index_names:
            index_names = [data.index.names[0]]

        reset_data = data.reset_index()
        key_cols = [*index_names, 'loc', 'dir']

        if reset_data.duplicated(key_cols).any():
            msg = (
                'Cannot form RID matrix because duplicate rows exist for '
                'the same non-location index, location, and direction.'
            )
            raise ValueError(msg)

        return reset_data.pivot_table(
            values='RID',
            index=index_names,
            columns=['loc', 'dir'],
            aggfunc='first',
        )

    @staticmethod
    def _sorted_index_values(data: pd.DataFrame, level_name: str) -> list[str]:
        return sorted(
            str(value) for value in data.index.get_level_values(level_name).unique()
        )

    @staticmethod
    def _sorted_column_values(data: pd.DataFrame, level_name: str) -> list[str]:
        return sorted(
            str(value)
            for value in data.columns.get_level_values(level_name).unique()
        )

    @staticmethod
    def _data_for_index_value(
        data: pd.DataFrame,
        level_name: str,
        level_value: str,
    ) -> pd.DataFrame:
        mask = data.index.get_level_values(level_name).astype(str) == level_value
        return data[mask]

    @staticmethod
    def _data_for_loc_dir(
        data: pd.DataFrame,
        location: str,
        direction: str,
    ) -> pd.DataFrame:
        loc_mask = data.index.get_level_values('loc').astype(str) == location
        dir_mask = data.index.get_level_values('dir').astype(str) == direction
        return data[loc_mask & dir_mask]

    def _validate_sample_input(self, pid: pd.DataFrame) -> None:
        if not self.fitted:
            msg = 'Residual drift model has not been fitted yet.'
            raise ValueError(msg)

        if not isinstance(pid.columns, pd.MultiIndex):
            msg = '`pid` must have MultiIndex columns.'
            raise TypeError(msg)

        if 'loc' not in pid.columns.names or 'dir' not in pid.columns.names:
            msg = '`pid` columns must include `loc` and `dir`.'
            raise ValueError(msg)

    def _sample_max_max(
        self,
        pid_df: pd.DataFrame,
        pid_directions: list[str],
    ) -> pd.DataFrame:
        if len(pid_directions) != EXPECTED_PID_COLUMN_COUNT:
            msg = (
                '`max_max` expects PID samples for exactly one direction at a time.'
            )
            raise ValueError(msg)

        direction = pid_directions[0]
        pid_max = pid_df.max(axis=1).to_numpy()
        model = self._max_max_model_for_direction(direction)
        rid_vals = model.generate_rid_samples(pid_max)

        return pd.DataFrame(
            rid_vals,
            index=pid_df.index,
            columns=pd.MultiIndex.from_tuples(
                [('RID', '0', direction)],
                names=['edp_type', 'loc', 'dir'],
            ),
        )

    def _max_max_model_for_direction(
        self, direction: str
    ) -> Model_Trilinear_Weibull:
        if self.fit_directions_together:
            assert self.model_maxmax is not None
            return self.model_maxmax

        if direction not in self.models_maxmax_by_dir:
            msg = f'No fitted max-max model available for direction `{direction}`.'
            raise ValueError(msg)
        return self.models_maxmax_by_dir[direction]

    def _check_sample_locations(self, pid_locations: list[str]) -> None:
        missing_locations = set(pid_locations) - set(self.locations)
        if missing_locations:
            msg = (
                'PID sample contains locations not present in the fitted model: '
                f'{sorted(missing_locations)}.'
            )
            raise ValueError(msg)

    def _sample_story(
        self,
        pid_df: pd.DataFrame,
        pid_locations: list[str],
        pid_directions: list[str],
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        l_mat = self._correlation_cholesky(pid_locations)
        pid_loc_level = pid_df.columns.names.index('loc')
        pid_dir_level = pid_df.columns.names.index('dir')
        rid_cols: list[pd.Series] = []

        for direction in pid_directions:
            u_corr = self._correlated_uniforms(
                rng, len(pid_df), len(pid_locations), l_mat
            )
            for i_loc, location in enumerate(pid_locations):
                rid_cols.append(
                    self._sample_story_location_direction(
                        pid_df,
                        location,
                        direction,
                        u_corr[:, i_loc],
                        pid_loc_level,
                        pid_dir_level,
                    )
                )

        rid = pd.concat(rid_cols, axis=1)
        rid.columns = pd.MultiIndex.from_tuples(
            rid.columns,
            names=['edp_type', 'loc', 'dir'],
        )
        return rid

    def _correlation_cholesky(self, pid_locations: list[str]) -> np.ndarray:
        assert self.correlation_matrix is not None
        corr_loc = self.correlation_matrix.reindex(
            index=pid_locations,
            columns=pid_locations,
        ).fillna(0.0)
        corr_values = self._nearest_psd(corr_loc.to_numpy())
        return np.linalg.cholesky(corr_values)

    @staticmethod
    def _correlated_uniforms(
        rng: np.random.Generator,
        n_samples: int,
        n_locations: int,
        cholesky_factor: np.ndarray,
    ) -> np.ndarray:
        u = rng.uniform(0.0, 1.0, size=(n_samples, n_locations))
        z = sp.stats.norm.ppf(u)
        z_corr = (cholesky_factor @ z.T).T
        return sp.stats.norm.cdf(z_corr)

    def _sample_story_location_direction(
        self,
        pid_df: pd.DataFrame,
        location: str,
        direction: str,
        uniform_sample: np.ndarray,
        pid_loc_level: int,
        pid_dir_level: int,
    ) -> pd.Series:
        pid_vals = self._pid_values_for_loc_dir(
            pid_df,
            location,
            direction,
            pid_loc_level,
            pid_dir_level,
        )
        model = self._story_model_for_location_direction(location, direction)
        model.uniform_sample = uniform_sample
        rid_vals = model.generate_rid_samples(pid_vals)
        return pd.Series(
            rid_vals,
            index=pid_df.index,
            name=('RID', location, direction),
        )

    @staticmethod
    def _pid_values_for_loc_dir(
        pid_df: pd.DataFrame,
        location: str,
        direction: str,
        pid_loc_level: int,
        pid_dir_level: int,
    ) -> np.ndarray:
        mask = [
            str(col[pid_loc_level]) == location
            and str(col[pid_dir_level]) == direction
            for col in pid_df.columns
        ]
        pid_series = pid_df.loc[:, mask]

        if pid_series.shape[1] != EXPECTED_PID_COLUMN_COUNT:
            msg = (
                'Expected exactly one PID column for '
                f'location `{location}` and direction `{direction}`.'
            )
            raise ValueError(msg)

        return pid_series.iloc[:, 0].to_numpy()

    def _story_model_for_location_direction(
        self,
        location: str,
        direction: str,
    ) -> Model_Trilinear_Weibull:
        if self.fit_directions_together:
            return self.models_by_loc[location]
        return self.models_by_loc_dir[location, direction]

    def _diagnostic_entries(self) -> list[tuple[str, Model_Trilinear_Weibull]]:
        if self.approach == 'max_max':
            return self._diagnostic_max_max_entries()
        return self._diagnostic_story_entries()

    def _diagnostic_max_max_entries(
        self,
    ) -> list[tuple[str, Model_Trilinear_Weibull]]:
        if self.fit_directions_together:
            assert self.model_maxmax is not None
            return [('max_max', self.model_maxmax)]

        return [
            (f'max_max | dir={direction}', model_i)
            for direction in self.directions
            if (model_i := self.models_maxmax_by_dir.get(direction)) is not None
        ]

    def _diagnostic_story_entries(self) -> list[tuple[str, Model_Trilinear_Weibull]]:
        if self.fit_directions_together:
            return [
                (f'loc={location}', model_i)
                for location in self.locations
                if (model_i := self.models_by_loc.get(location)) is not None
            ]

        return [
            (f'loc={location} | dir={direction}', model_i)
            for location in self.locations
            for direction in self.directions
            if (model_i := self.models_by_loc_dir.get((location, direction)))
            is not None
        ]

    @staticmethod
    def _create_diagnostic_axes(
        n_plots: int,
        figsize: tuple[float, float],
        max_cols: int,
    ) -> tuple[plt.Figure, np.ndarray]:
        n_cols = min(max_cols, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))
        return plt.subplots(
            n_rows,
            n_cols,
            figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
            squeeze=False,
        )

    def _plot_diagnostic_entry(
        self,
        ax: Axes,
        plot_title: str,
        model_i: Model_Trilinear_Weibull,
        *,
        rolling: bool,
        training: bool,
        model: bool,
        show_parameters: bool,
        parameter_format: str,
    ) -> None:
        model_i.plot_model(
            ax=ax,
            rolling=rolling,
            training=training,
            model=model,
        )
        if show_parameters:
            plot_title = self._append_parameters_to_title(
                plot_title,
                model_i,
                parameter_format,
            )
        self._decorate_diagnostic_axis(ax, plot_title, model_i)

    @staticmethod
    def _append_parameters_to_title(
        plot_title: str,
        model_i: Model_Trilinear_Weibull,
        parameter_format: str,
    ) -> str:
        parameter_lines = ResidualDriftTrilinearWeibullModel._parameter_lines(
            model_i,
            parameter_format,
        )
        if parameter_lines:
            return f'{plot_title}\n' + '\n'.join(parameter_lines)
        return plot_title

    @staticmethod
    def _parameter_lines(
        model_i: Model_Trilinear_Weibull,
        parameter_format: str,
    ) -> list[str]:
        if model_i.parameters is None:
            return []

        if model_i.parameter_names is not None:
            return [
                f'{name}={value:{parameter_format}}'
                for name, value in zip(
                    model_i.parameter_names,
                    model_i.parameters,
                    strict=False,
                )
            ]

        return [
            f'p{i}={value:{parameter_format}}'
            for i, value in enumerate(model_i.parameters)
        ]

    @staticmethod
    def _decorate_diagnostic_axis(
        ax: Axes,
        plot_title: str,
        model_i: Model_Trilinear_Weibull,
    ) -> None:
        ax.set_title(plot_title)
        ax.set_xlabel('PID')
        ax.set_ylabel('RID')

        if model_i.raw_pid is None or model_i.raw_rid is None:
            return

        max_pid = float(np.max(model_i.raw_pid))
        max_rid = float(np.max(model_i.raw_rid))
        ax.set_xlim(0.0, AXIS_PADDING_FACTOR * max_pid)
        ax.set_ylim(0.0, AXIS_PADDING_FACTOR * max_rid)
