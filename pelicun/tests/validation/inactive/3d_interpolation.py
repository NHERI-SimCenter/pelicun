"""
With this code we verify that scipy's `RegularGridInterpolator` does
what we expect.
Created: `Sat Jun  1 03:07:28 PM PDT 2024`

"""

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

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
