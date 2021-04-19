from typing import List

import numpy as np


def to_array_of_lists(a: np.ndarray) -> List:
    col_count = a.shape[1]
    result = [list(e) for e in zip(*(a[:, col_index] for col_index in range(col_count)))]
    return result
