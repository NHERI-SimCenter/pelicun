#
# Copyright (c) 2022 Leland Stanford Junior University
# Copyright (c) 2022 The Regents of the University of California
#
# This file is part of the SimCenter Backend Applications
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
# this file. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay
#
# ruff: noqa: N999

"""Performs k-Nearest Neighbors spatial interpolation and resampling."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


# Nearest Neighbors Regression
def NNR(  # noqa: N802, C901
    target: np.ndarray,
    source: np.ndarray,
    known_values: np.ndarray,
    sample_size: int = -1,
    n_neighbors: int = 4,
    weight: str = 'distance2',
    seed: int | None = None,
) -> np.ndarray:
    """Perform k-NN spatial interpolation and resampling.

    This function estimates values at a set of target locations based on known
    values at a set of source locations. It can operate in two modes:
    calculating a deterministic, weighted-average "expected value", or
    performing a probabilistic resampling to generate new realizations based on
    the neighbors' values and their distance from the target point.

    Parameters
    ----------
    target : np.ndarray
        Coordinates of the target points where values are to be estimated.
        Expected shape: `(n_targets, n_dimensions)`.

    source : np.ndarray
        Coordinates of the source points with known values.
        Expected shape: `(n_sources, n_dimensions)`.

    known_values : np.ndarray
        The known values at the source locations. The dimensionality determines
        if the source values are deterministic or a probabilistic sample. Use
        the 3D option even if there is only one feature.

        - If 2D `(n_sources, n_features)`: A single value for each feature.
        - If 3D `(n_sources, n_features, n_realizations)`: A sample of
          values (multiple realizations) for each feature.

    sample_size : int, optional
        Controls the operating mode and the number of output realizations.
        Defaults to -1.

        - If -1: Calculates the **expected value** by taking a
          weighted average of the neighbors' values. When working with a 3D
          array of known values, it provides an expected value for each
          realization of each feature.
        - If > 0: Performs **probabilistic resampling** to generate
          `sample_size` new realizations at each target point.

    n_neighbors : int, optional
        The number of nearest neighbors (`k`) to consider for each target
        point. Defaults to 4.

    weight : {'distance2', 'distance1', 'uniform'}, optional
        The method used to weigh the influence of each neighbor.
        Defaults to 'distance2'.

        - 'distance2': Inverse-square distance weighting (1/d^2).
        - 'distance1': Inverse distance weighting (1/d).
        - 'uniform': All neighbors are weighted equally.

    seed : int | None, optional
        A seed for the random number generator to ensure reproducibility of
        the probabilistic resampling (`sample_size > 0`). Defaults to None.

    Returns
    -------
    np.ndarray
        The estimated values at the target locations. The shape of the output
        depends on the `sample_size` parameter.

        - If `sample_size > 0`, the shape is
          `(n_targets, n_features, sample_size)`.
        - If `sample_size == -1` and `known_values` is 3D, the shape is
          `(n_targets, n_features, n_realizations)`.
        - If `sample_size == -1` and `known_values` is 2D, the shape is
          `(n_targets, n_features)`.

    """
    EPSILON = 1e-20  # noqa: N806
    if source.ndim == 1:
        source_locations = source.reshape(-1, 1)
        target_locations = target.reshape(-1, 1)
    else:
        source_locations = source.copy()
        target_locations = target.copy()

    source_values = known_values.copy()
    if source_values.ndim == 1:
        source_values = source_values.reshape(-1, 1)
    feature_count = source_values.shape[1]

    if sample_size > 0:
        predicted_values = np.full(
            (target_locations.shape[0], feature_count, sample_size), np.nan
        )
    elif source_values.ndim == 3:  # noqa: PLR2004
        n_realizations = source_values.shape[2]
        predicted_values = np.full(
            (target_locations.shape[0], feature_count, n_realizations), np.nan
        )
    else:
        predicted_values = np.full(
            (target_locations.shape[0], feature_count), np.nan
        )

    # prepare the tree for the nearest neighbor search
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(
        source_locations
    )

    # collect the neighbor indices and distances for every target point
    distances, indices = nbrs.kneighbors(target_locations)
    distances += EPSILON  # this is to avoid zero distance

    # initialize the random generator
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # iterate through the target points and store the sampled neighbors in an array
    for target_i, (dist_list, ind_list) in enumerate(zip(distances, indices)):
        if weight == 'distance2':
            dist_weights = 1.0 / (dist_list**2.0)
            weights = np.array(dist_weights) / np.sum(dist_weights)

        elif weight == 'uniform':
            weights = np.full(dist_list.shape, 1.0 / n_neighbors)

        elif weight == 'distance1':
            # calculate the weights for each neighbor based on their distance
            dist_weights = 1.0 / dist_list
            weights = np.array(dist_weights) / np.sum(dist_weights)

        if sample_size > 0:
            # get the sample of neighbor indices
            neighbor_sample = np.where(
                rng.multinomial(1, weights, sample_size) == 1
            )[1]

            val_list = np.full((feature_count, sample_size), np.nan)

            neighbor_set = np.unique(neighbor_sample)

            # for each unique neighbor
            for neighbor_rank in neighbor_set:
                # get the realizations from this neighbor
                neighbor_mask = np.where(neighbor_sample == neighbor_rank)[0]

                # get the index of the nth neighbor
                neighbor_id = ind_list[neighbor_rank]

                # make sure we resample values if sample_size > value_count
                if source_values.ndim == 3:  # noqa: PLR2004
                    value_j = neighbor_mask % source_values.shape[2]

                    # save the corresponding values
                    val_list[:, neighbor_mask] = source_values[
                        neighbor_id, :, value_j
                    ].T

                elif source_values.ndim == 2:  # noqa: PLR2004
                    val_list[:, neighbor_mask] = np.broadcast_to(
                        source_values[neighbor_id, :],
                        (neighbor_mask.size, feature_count),
                    ).T

                else:
                    val_list[:, neighbor_mask] = source_values[neighbor_id, :].T

        # if sample_size is -1, the expected value is returned
        elif source_values.ndim == 3:  # noqa: PLR2004
            # Get the values for all neighbors for all features and realizations
            neighbor_values = source_values[ind_list, :, :]

            # Reshape weights for broadcasting
            weights_3d = weights.reshape(-1, 1, 1)

            # Apply weights and sum along the neighbors axis in one step
            val_list = np.sum(weights_3d * neighbor_values, axis=0)

        elif source_values.ndim == 2:  # noqa: PLR2004
            val_list = weights @ source_values[ind_list, :]

        else:
            val_list = weights @ np.mean(source_values, axis=1)[ind_list].T

        predicted_values[target_i] = val_list

    if (known_values.ndim == 1) and (sample_size == -1):
        predicted_values = np.ravel(predicted_values)

    return predicted_values
