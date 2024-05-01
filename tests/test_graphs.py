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


def save_graph_figure(
    q_values: List[float],
    q_radii: List[float],
    lower_bounds: List[float],
    upper_bounds: List[float],
    output_path: Path,
):
    """
    This function accepts the approximated q-numerical radii together with
    their lower and upper bounds obtained via Inequality (18) or (19), and
    creates a figure where the three corresponding graphs are displayed via the
    functionalities from the `matplotlib.pyplot` package. The created figure is
    subsequently saved in the form of an image file.

    :arg q_values: A nonempty list of floats that represent the real q-values
        from the interval (0, 1] for which the q-numerical radius is
        approximated alongside the lower and upper bound.
    :arg q_radii: A nonempty list of floats representing the approximated
        q-numerical radii corresponding to the provided q-values, respectively.
        This list must have the same length as the `q_values` list.
    :arg lower_bounds: A nonempty list of floats representing the approximated
        lower bounds corresponding to the provided q-values, respectively. This
        list must have the same length as the `q_values` list.
    :arg upper_bounds: A nonempty list of floats representing the approximated
        upper bounds corresponding to the provided q-values, respectively. This
        list must have the same length as the `q_values` list.
    :arg output_path: A `Path` object that signifies the output path where the
        created figure should be saved.
    """

    plt.figure(figsize=(8, 6))

    # Plot all three graphs that correspond to the q-numerical radius alongside
    # its lower and upper bound.
    plt.plot(q_values, lower_bounds, label="lower bound")
    plt.plot(q_values, q_radii, label="q-numerical radius")
    plt.plot(q_values, upper_bounds, label="upper bound")

    # Set up the labels, legend and grid properly.
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(False)

    # Save the created figure at the provided file location.
    plt.savefig(str(output_path))
    plt.close("all")


def execute_test_1(
    matrix_1: np.ndarray,
    matrix_2: np.ndarray,
    q_values: List[float],
    iterations: int,
    output_folder: Path,
):
    """
    This function performs an experimental test of Inequality (18). More
    precisely, it takes two square matrices A and B of the same order and then
    approximates three distinct values for all the provided q-values:
    1. the q-numerical radius of the block diagonal matrix
        A O
        O B;
    2. the lower bound given in Inequality (18);
    3. the upper bound given in Inequality (18).
    All the q-numerical radii are approximated via Lemma 3.1. The graphical
    representation of the obtained results is then saved in the form of a PNG
    image with the help of the `save_graph_figure` function.

    :arg matrix_1: The input matrix A from Inequality (18), given in the form
        of a 2-dimensional `np.ndarray` object that represents any complex
        square matrix of order at least two.
    :arg matrix_2: The input matrix B from Inequality (18), given in the form
        of a 2-dimensional `np.ndarray` object that represents any complex
        square matrix of order at least two. This matrix must be of the same
        order as `matrix_1`.
    :arg q_values: A nonempty list of floats signifying the real numbers from
        the interval (0, 1] that represents the q-values for which Inequality
        (18) should be tested.
    :arg iterations: A positive integer which dictates how many random
        selections of feasible vectors should be made in total.
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

    # The lists containing the approximated q-numerical radii and their lower
    # and upper bounds should all be initialized to an empty list.
    q_radii = []
    lower_bounds = []
    upper_bounds = []

    # Iterate through all the possible q-values...
    for q_value in q_values:
        # Determine the q-numerical radius of matrix A.
        _, q_radius_1 = approximate_q_range(
            matrix=matrix_1,
            q_value=q_value,
            selection_function=orthogonal_selection,
            iterations=iterations,
        )
        # Determine the q-numerical radius of matrix B.
        _, q_radius_2 = approximate_q_range(
            matrix=matrix_2,
            q_value=q_value,
            selection_function=orthogonal_selection,
            iterations=iterations,
        )

        # Approximate the lower and upper bound from Inequality (18) and update
        # the corresponding lists.
        lower_bound = max(q_radius_1, q_radius_2)
        lower_bounds.append(lower_bound)
        upper_bound = (
            (q_value + 2 * np.sqrt(1 - q_value**2)) / q_value * lower_bound
        )
        upper_bounds.append(upper_bound)

        # Approximate the q-numerical radius of matrix [A O \\ O B] and update
        # the corresponding list.
        _, q_radius = approximate_q_range(
            matrix=matrix_3,
            q_value=q_value,
            selection_function=orthogonal_selection,
            iterations=iterations,
        )
        q_radii.append(q_radius)

    # Use the auxiliary `save_graph_figure` function in order to create the
    # desired Inequality (18) graphical representation and then save it in the
    # form of a PNG file.
    save_graph_figure(
        q_values=q_values,
        q_radii=q_radii,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        output_path=output_folder / "test_1.png",
    )


def execute_test_2(
    matrix_1: np.ndarray,
    matrix_2: np.ndarray,
    q_values: List[float],
    iterations: int,
    output_folder: Path,
):
    """
    This function performs an experimental test of Inequality (19). More
    precisely, it takes two square matrices A and B of the same order and then
    approximates three distinct values for all the provided q-values:
    1. the q-numerical radius of the block diagonal matrix
        A B
        B A;
    2. the lower bound given in Inequality (19);
    3. the upper bound given in Inequality (19).
    All the q-numerical radii are approximated via Lemma 3.1. The graphical
    representation of the obtained results is then saved in the form of a PNG
    image with the help of the `save_graph_figure` function.

    :arg matrix_1: The input matrix A from Inequality (19), given in the form
        of a 2-dimensional `np.ndarray` object that represents any complex
        square matrix of order at least two.
    :arg matrix_2: The input matrix B from Inequality (19), given in the form
        of a 2-dimensional `np.ndarray` object that represents any complex
        square matrix of order at least two. This matrix must be of the same
        order as `matrix_1`.
    :arg q_values: A nonempty list of floats signifying the real numbers from
        the interval (0, 1] that represents the q-values for which Inequality
        (19) should be tested.
    :arg iterations: A positive integer which dictates how many random
        selections of feasible vectors should be made in total.
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

    # The lists containing the approximated q-numerical radii and their lower
    # and upper bounds should all be initialized to an empty list.
    q_radii = []
    lower_bounds = []
    upper_bounds = []

    # Iterate through all the possible q-values...
    for q_value in q_values:
        # Determine the q-numerical radius of matrix A + B.
        _, q_radius_1 = approximate_q_range(
            matrix=matrix_1 + matrix_2,
            q_value=q_value,
            selection_function=orthogonal_selection,
            iterations=iterations,
        )
        # Determine the q-numerical radius of matrix A - B.
        _, q_radius_2 = approximate_q_range(
            matrix=matrix_1 - matrix_2,
            q_value=q_value,
            selection_function=orthogonal_selection,
            iterations=iterations,
        )

        # Approximate the lower and upper bound from Inequality (19) and update
        # the corresponding lists.
        lower_bound = max(q_radius_1, q_radius_2)
        lower_bounds.append(lower_bound)
        upper_bound = (
            (q_value + 2 * np.sqrt(1 - q_value**2)) / q_value * lower_bound
        )
        upper_bounds.append(upper_bound)

        # Approximate the q-numerical radius of matrix [A B \\ B A] and update
        # the corresponding list.
        _, q_radius = approximate_q_range(
            matrix=matrix_3,
            q_value=q_value,
            selection_function=orthogonal_selection,
            iterations=iterations,
        )
        q_radii.append(q_radius)

    # Use the auxiliary `save_graph_figure` function in order to create the
    # desired Inequality (19) graphical representation and then save it in the
    # form of a PNG file.
    save_graph_figure(
        q_values=q_values,
        q_radii=q_radii,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        output_path=output_folder / "test_2.png",
    )


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
        q_values=(0.4 + np.arange(61) / 100.0).tolist(),
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
        q_values=(0.4 + np.arange(61) / 100.0).tolist(),
        iterations=500000,
        output_folder=output_folder,
    )
