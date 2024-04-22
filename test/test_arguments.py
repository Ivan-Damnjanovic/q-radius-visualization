"""
This auxiliary module contains all the test arguments that should be used
by the `test_main` Python script.
"""

from typing import Dict, List

import numpy as np

# This is the only global variable defined in the entire module. It
# represents a list whose elements are dictionaries that contain the
# desired test arguments. Each dictionary has precisely three keys:
# 1. the value corresponding to the "matrix" key is the input matrix whose
# q-numerical range and q-numerical radius should be approximated;
# 2. the value corresponding to the "q-values" key is a list of complex
# numbers from the closed unit disc that signify the q-values for which
# the q-numerical range and q-numerical radius should be approximated;
# 3. the value corresponding to the "iterations" key is merely the number
# of random selections of feasible vectors to be made while the
# `approximate_q_range` function is being executed.
TEST_ARGUMENTS: List[Dict] = [
    {
        "matrix": np.array(
            [
                [7, 2, -5],
                [9, -19, 11],
                [0, 13, -1],
            ]
        ),
        "q_values": [0.73],
        "iterations": 1000000,
    },
    {
        "matrix": np.array(
            [
                [123 + 54j, 211 - 23j, 50],
                [37 - 372j, 170 - 11j, 20 + 230j],
                [50 + 77j, -113 + 29j, 251j],
            ]
        ),
        "q_values": [0.0, 0.2 + 0.43j, 1.0, -1.0, 1j],
        "iterations": 1000000,
    },
    {
        "matrix": np.array(
            [
                [123 + 54j, 211 - 23j],
                [37 - 372j, 170 - 11j],
            ]
        ),
        "q_values": [0.0, 0.2 + 0.43j, 1.0, -1.0, -1j],
        "iterations": 1000000,
    },
    {
        "matrix": np.array(
            [
                [123 + 54j, 211 - 23j, 50, 100],
                [37 - 372j, 170 - 11j, 20 + 230j, 120 - 58j],
                [50 + 77j, -113 + 29j, 251j, 20 + 117j],
                [70j, 203 - 88j, 111 + 54j, -73 + 28j],
            ]
        ),
        "q_values": [0.0, 0.42 - 0.11j, 1j],
        "iterations": 1000000,
    },
    {
        "matrix": np.array(
            [
                [-16.3, 28, -2.11],
                [17, 0.5, -5],
                [21.952, -12.8, 0],
            ]
        ),
        "q_values": (np.arange(11) / 10.0).tolist(),
        "iterations": 1000000,
    },
    {
        "matrix": np.array(
            [
                [5.0 + 2.3j, -1 + 3.5j, 8 - 12.11j],
                [7.2j, -5.66 + 25j, 0],
                [-12.3, 10 + 2j, -1.15 + 3j],
            ]
        ),
        "q_values": (np.arange(11) / 10.0).tolist(),
        "iterations": 1000000,
    },
    {
        "matrix": np.array(
            [
                [5, 2 + 3j, 17 - 8j],
                [2 - 3j, -11, 20 - 4j],
                [17 + 8j, 20 + 4j, 27],
            ]
        ),
        "q_values": (np.arange(11) / 10.0).tolist(),
        "iterations": 1000000,
    },
]
