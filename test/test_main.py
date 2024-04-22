"""
This is a short Python script that relies on the functionalities from the
`approximate_q_range` module in order to perform various test cases
regarding the approximation of q-numerical ranges and q-numerical radii. 
"""

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

# The test case arguments are imported from the auxiliary `test_arguments`
# Python module.
from test_arguments import TEST_ARGUMENTS

from q_radius_visualization.approximate_q_range import (
    approximate_q_range,
    naive_selection,
    orthogonal_selection,
)

# This global variable contains the output directory path and should be
# configured for each machine separately.
OUTPUT_PATH = r"D:\q-radius\output"


def save_comparison_figure(
    q_range_1: Union[ConvexHull, np.ndarray],
    q_range_2: Union[ConvexHull, np.ndarray],
    output_path: Path,
):
    """
    This function accepts two approximated q-numerical ranges and creates
    a figure where both of them are displayed one atop of the other via
    the functionalities from the `matplotlib.pyplot` package. Both ranges
    are displayed in such a way that only their boundary is colored, while
    the interior is not. The boundary of the first q-numerical range is
    colored in red, while the boundary of the second is colored in blue.
    The created figure is subsequently saved in the form of an image file.

    :arg q_range_1: The first input q-numerical range, given in the form
        of a `scipy.spatial.ConvexHull` or `np.ndarray` object.
    :arg q_range_2: The second input q-numerical range, given in the form
        of a `scipy.spatial.ConvexHull` or `np.ndarray` object.
    :arg output_path: A `Path` object that signifies the output path where
        the created figure should be saved.
    """

    # The 1:1 aspect ratio is always used.
    plt.gca().set_aspect("equal")

    if isinstance(q_range_1, ConvexHull):
        # If the first q-numerical range is represented in the form of a
        # `ConvexHull` object, then the points of this convex hull should
        # be selected and used in order to draw an approximative polygon.
        plt.fill(
            q_range_1.points[q_range_1.vertices, 0],
            q_range_1.points[q_range_1.vertices, 1],
            edgecolor="r",
            linewidth=3,
            fill=False,
        )
    else:
        # However, if the first q-numerical range is represented in the
        # form of an `np.ndarray` object, then all the corresponding
        # points should just be directly drawn as they are.
        plt.plot(
            q_range_1[:, 0],
            q_range_1[:, 1],
            color="r",
            linewidth=3,
        )

    # The second q-numerical range should be handled in the identical
    # manner.
    if isinstance(q_range_2, ConvexHull):
        plt.fill(
            q_range_2.points[q_range_2.vertices, 0],
            q_range_2.points[q_range_2.vertices, 1],
            edgecolor="b",
            linewidth=3,
            fill=False,
        )
    else:
        plt.plot(
            q_range_2[:, 0],
            q_range_2[:, 1],
            color="b",
            linewidth=3,
        )

    # Correct the plot display in case its width is too small.
    left, right = plt.xlim()
    if np.abs(right - left) < 5.0:
        plt.xlim(left - 2.5, right + 2.5)

    # Correct the plot display in case its height is too small.
    bottom, top = plt.ylim()
    if np.abs(top - bottom) < 5.0:
        plt.ylim(bottom - 2.5, top + 2.5)

    # Save the created figure at the provided file location.
    plt.savefig(str(output_path))
    plt.close("all")


def execute_test_case(
    matrix: np.ndarray,
    q_value: Union[complex, float],
    iterations: int,
    output_folder: Path,
    test_id: int,
    q_value_id: int,
):
    """
    This function executes a provided test case. More precisely, it takes
    an input matrix and q-value, and then approximately determines the
    corresponding q-numerical range and q-numerical radius of the given
    matrix. The provided number of feasible vector selections is performed
    and the approximation is done twice: with and without the help of
    Lemma 3.1. The obtained results are saved in two files:
    1. A text file which contains the approximated q-numerical radius,
    with and without the help of Lemma 3.1.
    2. A PNG file that contains the graphical representation of the
    approximated q-numerical range. The boundaries of two regions are
    displayed: the red boundary signifies the q-numerical range obtained
    without the use of Lemma 3.1, while the blue boundary describes the
    q-numerical range approximated via Lemma 3.1. Both of these files are
    named after two corresponding identifiers, the first of which denotes
    the test case dictionary index, while the second denotes the selected
    q-value index.

    :arg matrix: The input matrix whose q-numerical range and q-numerical
        radius should be approximated, given in the form of a
        2-dimensional `np.ndarray` object that represents any complex
        square matrix of order at least two.
    :arg q_value: A `complex` or a `float` signifying the complex number
        from the closed unit disc that represents the q-value for which
        the q-numerical range and q-numerical radius should be
        approximated.
    :arg iterations: A positive integer which dictates how many random
        selections of feasible vectors should be made in total.
    :arg output_folder: A `Path` object that signifies the folder where
        the two output files should be saved.
    :arg test_id: A nonnegative integer which represents the test case
        group identifier, i.e., the index of the test case dictionary that
        is being used.
    :arg q_value_id: A nonnegative integer that signifies the index of the
        selected q-value.
    """

    # Convert the `test_id` and `q_value_id` identifiers into a string
    # with precisely four characters. For example, 13 should get
    # transformed to "0013".
    test_id_string = str(test_id).zfill(4)
    q_value_id_string = str(q_value_id).zfill(4)

    # Determine the names for the two output files.
    output_log_name = f"{test_id_string}-{q_value_id_string}.txt"
    output_figure_name = f"{test_id_string}-{q_value_id_string}.png"

    with open(str(output_folder / output_log_name), "w") as opened_file:
        # Approximate the q-numerical range and q-numerical radius via the
        # `approximate_q_range` function without the help of Lemma 3.1.
        q_range_1, q_radius_1 = approximate_q_range(
            matrix=matrix,
            q_value=q_value,
            selection_function=naive_selection,
            iterations=iterations,
        )
        # Save the first approximated q-numerical radius to the output
        # text file.
        opened_file.write(
            f"Approximation process done via the naive selection "
            f"mechanism. The approximated q-numerical radius is "
            f"{q_radius_1:.4f}.\n"
        )
        # Approximate the q-numerical range and q-numerical radius via the
        # `approximate_q_range` function by relying on Lemma 3.1.
        q_range_2, q_radius_2 = approximate_q_range(
            matrix=matrix,
            q_value=q_value,
            selection_function=orthogonal_selection,
            iterations=iterations,
        )
        # Save the second approximated q-numerical radius to the output
        # text file.
        opened_file.write(
            f"Approximation process done via the orthogonal selection "
            f"mechanism. The approximated q-numerical radius is "
            f"{q_radius_2:.4f}.\n"
        )

    # Use the auxiliary `save_comparison_figure` function in order to
    # create the desired q-numerical range graphical representation and
    # then save it in the form of a PNG file.
    save_comparison_figure(
        q_range_1=q_range_1,
        q_range_2=q_range_2,
        output_path=output_folder / output_figure_name,
    )


# This is the entry point of the script.
if __name__ == "__main__":
    # All the test results should be saved in the output folder. If such a
    # directory does not exist, then it should get created first.
    output_folder = Path(OUTPUT_PATH)
    output_folder.mkdir(exist_ok=True)

    # Iterate through all the test case dictionaries...
    for test_id, test_arguments in enumerate(TEST_ARGUMENTS):
        # Iterate through all the possible q-values corresponding to the
        # given input matrix...
        for q_value_id, q_value in enumerate(test_arguments["q_values"]):
            # Execute each test case separately.
            execute_test_case(
                matrix=test_arguments["matrix"],
                q_value=q_value,
                iterations=test_arguments["iterations"],
                output_folder=output_folder,
                test_id=test_id,
                q_value_id=q_value_id,
            )
