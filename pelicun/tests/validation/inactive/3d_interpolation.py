# noqa: N999
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
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

"""
With this code we verify that scipy's `RegularGridInterpolator` does
what we expect.

"""

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def main():
    # Define domains
    num_samples = 100
    dom1 = np.linspace(0, 1, num_samples)
    dom2 = np.linspace(0, 1, num_samples)
    dom3 = np.linspace(0, 1, num_samples)

    # Define 3D array
    vg1, vg2, vg3 = np.meshgrid(dom1, dom2, dom3)
    values = vg1 + np.sqrt(vg2) + np.sin(vg3)

    # Define test inputs for interpolation.
    x1 = np.random.rand(10)
    x2 = np.random.rand(10)
    x3 = np.random.rand(10)
    test_values = np.column_stack((x1, x2, x3))

    # Create the interpolation function
    interp_func = RegularGridInterpolator((dom1, dom2, dom3), values)

    # Perform the interpolation
    interpolated_value = interp_func(test_values)

    # Compare output with the exact value.
    df = pd.DataFrame(
        {
            'exact': x1 + np.sqrt(x2) + np.sin(x3),
            'interpolated': interpolated_value,
        }
    )
    print(df)

    # Note: This does work with a 2D case, and it could scale to more than
    # 3 dimensions.


if __name__ == '__main__':
    main()
