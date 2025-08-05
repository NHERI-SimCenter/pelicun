# -*- coding: utf-8 -*-
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

import numpy as np
from sklearn.neighbors import NearestNeighbors


# Nearest Neighbors Regression
def NNR(
    target,
    source,
    known_values,
    sample_size=-1,
    n_neighbors=4,
    weight='distance2',
    seed=None,
):
    if source.ndim == 1:
        X = source.reshape(-1, 1)
        X_t = target.reshape(-1, 1)
    else:
        X = source.copy()
        X_t = target.copy()

    Z = known_values.copy()
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    feature_count = Z.shape[1]

    if sample_size > 0:
        predicted_values = np.full(
            (X_t.shape[0], feature_count, sample_size), np.nan
        )
    else:
        predicted_values = np.full((X_t.shape[0], feature_count), np.nan)
    Z_hat = predicted_values

    # prepare the tree for the nearest neighbor search
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)

    # collect the neighbor indices and distances for every target point
    distances, indices = nbrs.kneighbors(X_t)
    distances = distances + 1e-20  # this is to avoid zero distance

    # initialize the random generator
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # iterate through the target points and store the sampled neighbors in an array
    for target_i, (dist_list, ind_list) in enumerate(zip(distances, indices)):
        if weight == 'distance2':
            # calculate the weights for each neighbor based on their squared distance
            dist_list = 1.0 / (dist_list**2.0)
            weights = np.array(dist_list) / np.sum(dist_list)

        elif weight == 'uniform':
            weights = np.full(dist_list.shape, 1.0 / n_neighbors)

        elif weight == 'distance1':
            # calculate the weights for each neighbor based on their distance
            dist_list = 1.0 / dist_list
            weights = np.array(dist_list) / np.sum(dist_list)

        if sample_size > 0:
            # get the pre-defined number of samples for each neighbor
            nbr_samples = np.where(rng.multinomial(1, weights, sample_size) == 1)[1]

            val_list = np.full((feature_count, sample_size), np.nan)

            nbr_unique = np.unique(nbr_samples)

            # for each unique neighbor
            for sample_j, nbr in enumerate(nbr_unique):
                # get the realizations from this neighbor
                nbr_mask = np.where(nbr_samples == nbr)[0]

                # get the index of the nth neighbor
                nbr_index = ind_list[nbr]

                # make sure we resample values if sample_size > value_count
                if Z.ndim == 3:
                    value_j = nbr_mask % Z.shape[2]

                    # save the corresponding values
                    val_list[:, nbr_mask] = Z[nbr_index, :, value_j].T

                elif Z.ndim == 2:
                    val_list[:, nbr_mask] = np.broadcast_to(
                        Z[nbr_index, :], (nbr_mask.size, feature_count)
                    ).T

                else:
                    val_list[:, nbr_mask] = Z[nbr_index, :].T

        # if sample_size is -1, the expected value is returned
        else:
            if Z.ndim == 3:
                val_list = []
                for feature_i in range(feature_count):
                    val_list.append(
                        weights @ np.mean(Z[ind_list, feature_i, :], axis=1).T
                    )

            elif Z.ndim == 2:
                val_list = []
                for feature_i in range(feature_count):
                    val_list.append(weights @ Z[ind_list, feature_i].T)

            else:
                val_list = weights @ np.mean(Z, axis=1)[ind_list].T

        Z_hat[target_i] = val_list

    if (known_values.ndim == 1) and (sample_size == -1):
        Z_hat = np.ravel(Z_hat)

    return Z_hat
