"""
This is a short Python script that relies on the functionalities from the
`approximate_q_range` module in order to experimentally demonstrate
Inequalities (18) and (19) for concrete pairs of A and B square matrices.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from q_radius_visualization.approximate_q_range import (
    approximate_q_range,
    orthogonal_selection,
)

# This global variable contains the output directory path and should be
# configured for each machine separately.
OUTPUT_PATH = r"../test_graphs_output"


def execute_test_1(
    matrix_1: np.ndarray,
    matrix_2: np.ndarray,
    analytical_q_values: List[float],
    numerical_q_values: List[float],
    iterations: int,
    output_folder: Path,
):
    """
    This function performs an experimental test of Inequality (18). More
    precisely, it takes two Hermitian matrices A and B of the same order and
    then determines three distinct values for all the provided q-values:
    1. the q-numerical radius of the block diagonal matrix
        A O
        O B;
    2. the lower bound given in Inequality (18);
    3. the upper bound given in Inequality (18).
    Each of these three values is computed in two ways:
    1. directly via Expression (20);
    2. approximately via the `approximate_q_range` function by using Lemma 3.1.
    The function subsequently creates a figure where the six corresponding
    graphs are displayed via the functionalities from the `matplotlib.pyplot`
    package. The created figure is then saved in the form of a PNG file.

    :arg matrix_1: The input matrix A from Inequality (18), given in the form
        of a 2-dimensional `np.ndarray` object that represents any Hermitian
        matrix of order at least two.
    :arg matrix_2: The input matrix B from Inequality (18), given in the form
        of a 2-dimensional `np.ndarray` object that represents any Hermitian
        matrix of order at least two. This matrix must be of the same order as
        `matrix_1`.
    :arg analytical_q_values: A nonempty list of floats signifying the real
        numbers from the interval (0, 1] that represent the q-values for which
        the computations should be done directly via Expression (20).
    :arg numerical_q_values: A nonempty list of floats signifying the real
        numbers from the interval (0, 1] that represent the q-values for which
        the computations should be done approximately via the
        `approximate_q_range` function by using Lemma 3.1.
    :arg iterations: A positive integer which dictates how many random
        selections of feasible vectors should be made in total while the
        `approximate_q_range` function is being used.
    :arg output_folder: A `Path` object that signifies the folder where the
        output PNG file should be saved.
    """

    # Let `matrix_3` be the [A O \\ O B] matrix.
    shape_1 = matrix_1.shape[0]
    shape_2 = matrix_2.shape[0]
    matrix_3 = np.block(
        [
            [matrix_1, np.zeros((shape_1, shape_2))],
            [np.zeros((shape_2, shape_1)), matrix_2],
        ]
    )

    # Compute the eigenvalues of matrices A, B and [A O \\ O B].
    eigenvalues_1, _ = np.linalg.eigh(matrix_1)
    eigenvalues_2, _ = np.linalg.eigh(matrix_2)
    eigenvalues_3, _ = np.linalg.eigh(matrix_3)

    # The lists containing the analytically derived q-numerical radii and their
    # lower and upper bounds should all be initialized to an empty list.
    analytical_middle_terms = []
    analytical_lower_bounds = []
    analytical_upper_bounds = []

    # Iterate through all the possible q-values...
    for q_value in analytical_q_values:
        # Determine all the required q-numerical radii via Expression (20).
        q_radius_1 = 0.5 * (
            q_value * np.abs(eigenvalues_1[0] + eigenvalues_1[-1])
            + eigenvalues_1[-1]
            - eigenvalues_1[0]
        )
        q_radius_2 = 0.5 * (
            q_value * np.abs(eigenvalues_2[0] + eigenvalues_2[-1])
            + eigenvalues_2[-1]
            - eigenvalues_2[0]
        )
        q_radius_3 = 0.5 * (
            q_value * np.abs(eigenvalues_3[0] + eigenvalues_3[-1])
            + eigenvalues_3[-1]
            - eigenvalues_3[0]
        )

        # Compute the lower and upper bound from Inequality (18) and update the
        # corresponding lists.
        lower_bound = max(q_radius_1, q_radius_2)
        analytical_lower_bounds.append(lower_bound)
        upper_bound = (
            (q_value + 2 * np.sqrt(1 - q_value**2)) / q_value * lower_bound
        )
        analytical_upper_bounds.append(upper_bound)
        analytical_middle_terms.append(q_radius_3)

    # The lists containing the numerically approximated q-numerical radii and
    # their lower and upper bounds should all be initialized to an empty list.
    numerical_middle_terms = []
    numerical_lower_bounds = []
    numerical_upper_bounds = []

    # Iterate through all the possible q-values...
    for q_value in numerical_q_values:
        # Approximate the q-numerical radius of matrix A.
        _, q_radius_1 = approximate_q_range(
            matrix=matrix_1,
            q_value=q_value,
            selection_function=orthogonal_selection,
            iterations=iterations,
        )
        # Approximate the q-numerical radius of matrix B.
        _, q_radius_2 = approximate_q_range(
            matrix=matrix_2,
            q_value=q_value,
            selection_function=orthogonal_selection,
            iterations=iterations,
        )

        # Approximate the lower and upper bound from Inequality (18) and update
        # the corresponding lists.
        lower_bound = max(q_radius_1, q_radius_2)
        numerical_lower_bounds.append(lower_bound)
        upper_bound = (
            (q_value + 2 * np.sqrt(1 - q_value**2)) / q_value * lower_bound
        )
        numerical_upper_bounds.append(upper_bound)

        # Approximate the q-numerical radius of matrix [A O \\ O B] and update
        # the corresponding list.
        _, q_radius = approximate_q_range(
            matrix=matrix_3,
            q_value=q_value,
            selection_function=orthogonal_selection,
            iterations=iterations,
        )
        numerical_middle_terms.append(q_radius)

    # Initialize the figure size.
    plt.figure(figsize=(8, 6))

    # Plot all three graphs that correspond to the analytically derived
    # q-numerical radius alongside its lower and upper bound.
    plt.plot(
        analytical_q_values,
        analytical_lower_bounds,
        label="Lower bound from Eq. (18) [Analytical]",
    )
    plt.plot(
        analytical_q_values,
        analytical_middle_terms,
        label="Middle term from Eq. (18) [Analytical]",
    )
    plt.plot(
        analytical_q_values,
        analytical_upper_bounds,
        label="Upper bound from Eq. (18) [Analytical]",
    )
    # Plot all three graphs that correspond to the numerically approximated
    # q-numerical radius alongside its lower and upper bound.
    plt.plot(
        numerical_q_values,
        numerical_lower_bounds,
        label="Lower bound from Eq. (18) [Numerical]",
    )
    plt.plot(
        numerical_q_values,
        numerical_middle_terms,
        label="Middle term from Eq. (18) [Numerical]",
    )
    plt.plot(
        numerical_q_values,
        numerical_upper_bounds,
        label="Upper bound from Eq. (18) [Numerical]",
    )

    # Set up the label, legend and grid properly.
    plt.xlabel("q-value")
    plt.legend()
    plt.grid(False)

    # Save the created figure at the provided file location.
    plt.savefig(str(output_folder / "test_1.png"))
    plt.close("all")


def execute_test_2(
    matrix_1: np.ndarray,
    matrix_2: np.ndarray,
    analytical_q_values: List[float],
    numerical_q_values: List[float],
    iterations: int,
    output_folder: Path,
):
    """
    This function performs an experimental test of Inequality (19). More
    precisely, it takes two Hermitian matrices A and B of the same order and
    then determines three distinct values for all the provided q-values:
    1. the q-numerical radius of the block diagonal matrix
        A B
        B A;
    2. the lower bound given in Inequality (19);
    3. the upper bound given in Inequality (19).
    Each of these three values is computed in two ways:
    1. directly via Expression (20);
    2. approximately via the `approximate_q_range` function by using Lemma 3.1.
    The function subsequently creates a figure where the six corresponding
    graphs are displayed via the functionalities from the `matplotlib.pyplot`
    package. The created figure is then saved in the form of a PNG file.

    :arg matrix_1: The input matrix A from Inequality (19), given in the form
        of a 2-dimensional `np.ndarray` object that represents any Hermitian
        matrix of order at least two.
    :arg matrix_2: The input matrix B from Inequality (19), given in the form
        of a 2-dimensional `np.ndarray` object that represents any Hermitian
        matrix of order at least two. This matrix must be of the same order as
        `matrix_1`.
    :arg analytical_q_values: A nonempty list of floats signifying the real
        numbers from the interval (0, 1] that represent the q-values for which
        the computations should be done directly via Expression (20).
    :arg numerical_q_values: A nonempty list of floats signifying the real
        numbers from the interval (0, 1] that represent the q-values for which
        the computations should be done approximately via the
        `approximate_q_range` function by using Lemma 3.1.
    :arg iterations: A positive integer which dictates how many random
        selections of feasible vectors should be made in total while the
        `approximate_q_range` function is being used.
    :arg output_folder: A `Path` object that signifies the folder where the
        output PNG file should be saved.
    """

    # Let `matrix_3` be the [A B \\ B A] matrix.
    matrix_3 = np.block(
        [
            [matrix_1, matrix_2],
            [matrix_2, matrix_1],
        ]
    )

    # Compute the eigenvalues of matrices A + B, A - B and [A B \\ B A].
    eigenvalues_1, _ = np.linalg.eigh(matrix_1 + matrix_2)
    eigenvalues_2, _ = np.linalg.eigh(matrix_1 - matrix_2)
    eigenvalues_3, _ = np.linalg.eigh(matrix_3)

    # The lists containing the analytically derived q-numerical radii and their
    # lower and upper bounds should all be initialized to an empty list.
    analytical_middle_terms = []
    analytical_lower_bounds = []
    analytical_upper_bounds = []

    # Iterate through all the possible q-values...
    for q_value in analytical_q_values:
        # Determine all the required q-numerical radii via Expression (20).
        q_radius_1 = 0.5 * (
            q_value * np.abs(eigenvalues_1[0] + eigenvalues_1[-1])
            + eigenvalues_1[-1]
            - eigenvalues_1[0]
        )
        q_radius_2 = 0.5 * (
            q_value * np.abs(eigenvalues_2[0] + eigenvalues_2[-1])
            + eigenvalues_2[-1]
            - eigenvalues_2[0]
        )
        q_radius_3 = 0.5 * (
            q_value * np.abs(eigenvalues_3[0] + eigenvalues_3[-1])
            + eigenvalues_3[-1]
            - eigenvalues_3[0]
        )

        # Compute the lower and upper bound from Inequality (19) and update the
        # corresponding lists.
        lower_bound = max(q_radius_1, q_radius_2)
        analytical_lower_bounds.append(lower_bound)
        upper_bound = (
            (q_value + 2 * np.sqrt(1 - q_value**2)) / q_value * lower_bound
        )
        analytical_upper_bounds.append(upper_bound)
        analytical_middle_terms.append(q_radius_3)

    # The lists containing the numerically approximated q-numerical radii and
    # their lower and upper bounds should all be initialized to an empty list.
    numerical_middle_terms = []
    numerical_lower_bounds = []
    numerical_upper_bounds = []

    # Iterate through all the possible q-values...
    for q_value in numerical_q_values:
        # Approximate the q-numerical radius of matrix A + B.
        _, q_radius_1 = approximate_q_range(
            matrix=matrix_1 + matrix_2,
            q_value=q_value,
            selection_function=orthogonal_selection,
            iterations=iterations,
        )
        # Approximate the q-numerical radius of matrix A - B.
        _, q_radius_2 = approximate_q_range(
            matrix=matrix_1 - matrix_2,
            q_value=q_value,
            selection_function=orthogonal_selection,
            iterations=iterations,
        )

        # Approximate the lower and upper bound from Inequality (19) and update
        # the corresponding lists.
        lower_bound = max(q_radius_1, q_radius_2)
        numerical_lower_bounds.append(lower_bound)
        upper_bound = (
            (q_value + 2 * np.sqrt(1 - q_value**2)) / q_value * lower_bound
        )
        numerical_upper_bounds.append(upper_bound)

        # Approximate the q-numerical radius of matrix [A B \\ B A] and update
        # the corresponding list.
        _, q_radius = approximate_q_range(
            matrix=matrix_3,
            q_value=q_value,
            selection_function=orthogonal_selection,
            iterations=iterations,
        )
        numerical_middle_terms.append(q_radius)

    # Initialize the figure size.
    plt.figure(figsize=(8, 6))

    # Plot all three graphs that correspond to the analytically derived
    # q-numerical radius alongside its lower and upper bound.
    plt.plot(
        analytical_q_values,
        analytical_lower_bounds,
        label="Lower bound from Eq. (19) [Analytical]",
    )
    plt.plot(
        analytical_q_values,
        analytical_middle_terms,
        label="Middle term from Eq. (19) [Analytical]",
    )
    plt.plot(
        analytical_q_values,
        analytical_upper_bounds,
        label="Upper bound from Eq. (19) [Analytical]",
    )
    # Plot all three graphs that correspond to the numerically approximated
    # q-numerical radius alongside its lower and upper bound.
    plt.plot(
        numerical_q_values,
        numerical_lower_bounds,
        label="Lower bound from Eq. (19) [Numerical]",
    )
    plt.plot(
        numerical_q_values,
        numerical_middle_terms,
        label="Middle term from Eq. (19) [Numerical]",
    )
    plt.plot(
        numerical_q_values,
        numerical_upper_bounds,
        label="Upper bound from Eq. (19) [Numerical]",
    )

    # Set up the label, legend and grid properly.
    plt.xlabel("q-value")
    plt.legend()
    plt.grid(False)

    # Save the created figure at the provided file location.
    plt.savefig(str(output_folder / "test_2.png"))
    plt.close("all")


# This is the entry point of the script.
if __name__ == "__main__":
    # All the test results should be saved to the output folder. If such a
    # directory does not exist, then it should get created first.
    output_folder = Path(OUTPUT_PATH)
    output_folder.mkdir(exist_ok=True)

    # Perform an experimental test of Inequality (18) by using the
    # `execute_test_1` function.
    execute_test_1(
        matrix_1=np.array(
            [
                [-4, 2],
                [2, -1],
            ]
        ),
        matrix_2=np.array(
            [
                [3, 2],
                [2, 3],
            ]
        ),
        analytical_q_values=(0.4 + np.arange(61) / 100.0).tolist(),
        numerical_q_values=(0.4 + np.arange(61) / 100.0).tolist(),
        iterations=500000,
        output_folder=output_folder,
    )

    # Perform an experimental test of Inequality (19) by using the
    # `execute_test_2` function.
    execute_test_2(
        matrix_1=np.array(
            [
                [-4, 2],
                [2, -1],
            ]
        ),
        matrix_2=np.array(
            [
                [3, 2],
                [2, 3],
            ]
        ),
        analytical_q_values=(0.4 + np.arange(61) / 100.0).tolist(),
        numerical_q_values=(0.4 + np.arange(61) / 100.0).tolist(),
        iterations=500000,
        output_folder=output_folder,
    )
