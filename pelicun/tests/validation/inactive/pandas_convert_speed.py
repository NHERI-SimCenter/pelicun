import pandas as pd
import numpy as np
import time

# pylint: disable=pointless-statement


def benchmark():
    # Create a large DataFrame
    df = pd.DataFrame(np.random.rand(1000000, 10), columns=list('ABCDEFGHIJ'))

    # Measure time for df.to_dict(orient='list')
    start_time = time.time()
    df.to_dict(orient='list')
    end_time = time.time()
    print(f'Time taken with to_dict(orient="list"): {end_time - start_time} seconds')

    # Measure time for dictionary comprehension
    start_time = time.time()
    {col: df[col].tolist() for col in df.columns}
    end_time = time.time()
    print(
        f'Time taken with dictionary comprehension: {end_time - start_time} seconds'
    )

    # Measure time for dictionary comprehension without to list
    start_time = time.time()
    {col: df[col] for col in df.columns}
    end_time = time.time()
    print(
        f'Time taken with dictionary comprehension '
        f'without to_list: {end_time - start_time} seconds'
    )

    # Measure time for .values
    start_time = time.time()
    df.values
    end_time = time.time()
    print(f'Time taken with .values: {end_time - start_time} seconds')

    # Measure time for using df.to_numpy()
    start_time = time.time()
    data_array = df.to_numpy()
    {col: data_array[:, i].tolist() for i, col in enumerate(df.columns)}
    end_time = time.time()
    print(f'Time taken with df.to_numpy(): {end_time - start_time} seconds')

    # Measure time for using df.to_dict()
    start_time = time.time()
    df.to_dict()
    end_time = time.time()
    print(f'Time taken with df.to_dict(): {end_time - start_time} seconds')


if __name__ == '__main__':
    benchmark()
