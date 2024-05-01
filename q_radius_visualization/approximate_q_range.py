"""
This Python module contains several functions that deal with the approximation
of the q-numerical range and q-numerical radius of complex square matrices of
order at least two.
"""

from typing import Callable, Tuple, Union

import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import uniform_direction, unitary_group


def naive_selection(
    matrix: np.ndarray, q_value: Union[complex, float]
) -> Tuple[complex, float]:
    """
    For a given complex square matrix A and a q-value from the closed unit
    disc, this function makes a random selection of two unit vectors x and y
    such that <x, y> = q, and then computes the <Ax, y> inner product which
    surely belongs to the corresponding q-numerical range of A. The modulus of
    this value is also computed, thus yielding a lower bound for the
    q-numerical radius of A.

    :arg matrix: The input matrix A, given in the form of a 2-dimensional
        `np.ndarray` object that represents any complex square matrix of order
        at least two.
    :arg q_value: A `complex` or a `float` signifying the complex number from
        the closed unit disc that represents the q-value.

    :return: A tuple consisting of two items:
        1. The computed <Ax, y> inner product;
        2. The modulus of the obtained inner product, i.e., |<Ax, y>|.
    """

    # Check to make sure that the provided matrix is a complex square matrix of
    # order at least two.
    assert isinstance(matrix, np.ndarray)
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1] and matrix.shape[0] >= 2

    # Check to make sure that `q_value` is a `complex` or a `float` from the
    # closed unit disc.
    assert (
        isinstance(q_value, complex) or isinstance(q_value, float)
    ) and np.abs(q_value) <= 1.0

    # Let `order` contain the matrix order, which is at least two.
    order = matrix.shape[0]

    # Let `unitary_matrix` contain a randomly generated unitary matrix of order
    # `order` that is obtained via the `unitary_group` function.
    unitary_matrix = unitary_group(order).rvs(0)
    # Select the first column of the generated unitary matrix to be the y
    # vector. Therefore, this vector is just a randomly generated unit vector.
    y_vector = unitary_matrix[:, 0]

    # The `uniform_direction` function is used to obtain a randomly generated
    # unit real vector of order 2 * `order` - 2, which is then transformed into
    # a unit complex vector of order `order` - 1.
    aux_vector_1 = uniform_direction(2 * order - 2).rvs(1)[0]
    aux_vector_2 = aux_vector_1[: order - 1] + aux_vector_1[order - 1 :] * 1j

    # Let B be the base comprising the column vectors of `unitary_matrix`. The
    # x vector is formed in such a way that its coordinate corresponding to the
    # first column, i.e., the y vector, equals q, while the other coordinates
    # are obtained from the previously computed unit complex vector of order
    # `order` - 1. The necessary scaling is applied so that the x vector is
    # also a unit vector. Therefore, we have that x and y are randomly
    # generated unit complex vectors such that <x, y> = q.
    aux_vector_3 = np.hstack(
        [[q_value], np.sqrt(1 - np.abs(q_value) ** 2) * aux_vector_2]
    )
    x_vector = unitary_matrix @ aux_vector_3

    # Let `q_range_value` contain the <Ax, y> value, which certainly belongs to
    # the q-numerical range of the provided matrix. Moreover, let
    # `modulus_value` be the modulus of this value. It is guaranteed that the
    # q-numerical radius is greater than or equal to this number.
    q_range_value = np.dot(matrix @ x_vector, np.conj(y_vector))
    q_range_value = complex(q_range_value.item())
    modulus_value = float(np.abs(q_range_value).item())

    return q_range_value, modulus_value


def orthogonal_selection(
    matrix: np.ndarray, q_value: Union[complex, float]
) -> Tuple[complex, float]:
    """
    For a given complex square matrix A and a q-value from the closed unit
    disc, this function makes a random selection of two orthogonal unit vectors
    y and t, then computes the value q * <Ay, y> + sqrt(1 - |q|^2) * <At, y>
    which necessarily belongs to the q-numerical range of A, by virtue of
    Lemma 3.1. The number
    |q| * |<Ay, y>| + sqrt(1 - |q|^2) * |<At, y>|
    is also determined, thus yielding a lower bound for the q-numerical radius
    of A, again due to Lemma 3.1.

    :arg matrix: The input matrix A, given in the form of a 2-dimensional
        `np.ndarray` object that represents any complex square matrix of order
        at least two.
    :arg q_value: A `complex` or a `float` signifying the complex number from
        the closed unit disc that represents the q-value.

    :return: A tuple consisting of two items:
        1. The computed complex number that surely belongs to the
        q-numerical range of A;
        2. The determined lower bound for the q-numerical radius of A.
    """

    # Check to make sure that the provided matrix is a complex square matrix of
    # order at least two.
    assert isinstance(matrix, np.ndarray)
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1] and matrix.shape[0] >= 2

    # Check to make sure that `q_value` is a `complex` or a `float` from the
    # closed unit disc.
    assert (
        isinstance(q_value, complex) or isinstance(q_value, float)
    ) and np.abs(q_value) <= 1.0

    # Let `order` contain the matrix order, which is at least two.
    order = matrix.shape[0]

    # Let `unitary_matrix` contain a randomly generated unitary matrix of order
    # `order` that is obtained via the `unitary_group` function.
    unitary_matrix = unitary_group(order).rvs(0)
    # Select the first column of the generated unitary matrix to be the y
    # vector and the second column to be the t vector. Therefore, these two
    # vectors are just two randomly generated orthogonal unit vectors.
    y_vector = unitary_matrix[0, :]
    t_vector = unitary_matrix[1, :]

    # Let `first_term` contain the q * <Ay, y> value and let `second_term` be
    # the number obtained from the sqrt(1 - |q|^2) * <At, y> expression.
    first_term = q_value * np.dot(matrix @ y_vector, np.conj(y_vector))
    second_term = np.sqrt(1 - np.abs(q_value) ** 2) * np.dot(
        matrix @ t_vector, np.conj(y_vector)
    )

    # Let `q_range_value` contain the sum of `first_term` and `second_term`,
    # which certainly belongs to the q-numerical range of the provided matrix,
    # by virtue of Lemma 3.1.
    q_range_value = first_term + second_term
    q_range_value = complex(q_range_value.item())
    # Due to Lemma 3.1, we also know that the sum of moduli of `first_term` and
    # `second_term` is certainly a lower bound for the q-numerical radius.
    sum_of_moduli = np.abs(first_term) + np.abs(second_term)
    sum_of_moduli = float(sum_of_moduli.item())

    return q_range_value, sum_of_moduli


def approximate_q_range(
    matrix: np.ndarray,
    q_value: Union[complex, float],
    selection_function: Callable,
    iterations: int,
) -> Tuple[Union[ConvexHull, np.ndarray], float]:
    """
    This function takes a complex square matrix A and approximates its
    q-numerical range and q-numerical radius for the given q-value. The
    approximation is performed by randomly selecting a feasible pair of x and y
    vectors and by computing their according <Ax, y> inner product. This
    selection is made a necessary number of times and, bearing in mind that the
    q-numerical range is necessarily convex, we may approximate it by simply
    using the convex hull of all the obtained <Ax, y> complex points.

    :arg matrix: The input matrix A whose q-numerical range and q-numerical
        radius should be approximated, given in the form of a 2-dimensional
        `np.ndarray` object that represents any complex square matrix of order
        at least two.
    :arg q_value: A `complex` or a `float` signifying the complex number from
        the closed unit disc that represents the q-value for which the
        q-numerical range and q-numerical radius should be approximated.
    :arg selection_function: A function that randomly selects a feasible pair
        of x and y vectors and computes their <Ax, y> inner product, as well as
        the greatest |<Ax, y>| value which they guarantee. This function should
        accept two arguments: the input matrix and the q-value, and it should
        return two values: the inner product and its according obtained
        modulus.
    :arg iterations: A positive integer which dictates how many random
        selections of feasible x and y vectors should be made in total.

    :return: A tuple consisting of two items:
        1. The approximated q-numerical range, given either in the form of a
        `scipy.spatial.ConvexHull` object (if the generated points have a
        nondegenerate convex hull) or a 2-dimensional `np.ndarray` object
        (otherwise). If a `np.ndarray` matrix is returned, then it has two
        columns, with its rows representing the points that surely belong to
        the approximated q-numerical range.
        2. The approximated q-numerical radius, given the form of a float.
    """

    # Check to make sure that the provided matrix is a complex square matrix of
    # order at least two.
    assert isinstance(matrix, np.ndarray)
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1] and matrix.shape[0] >= 2

    # Check to make sure that `q_value` is a `complex` or a `float` from the
    # closed unit disc. Also, verify that the remaining arguments are valid as
    # well.
    assert (
        isinstance(q_value, complex) or isinstance(q_value, float)
    ) and np.abs(q_value) <= 1.0
    assert callable(selection_function)
    assert isinstance(iterations, int) and iterations >= 1

    # The q-numerical range should be initialized to an empty list, while the
    # q-numerical radius can be initialized to any negative value.
    q_range = []
    q_radius = -1

    count = 0
    # Perform the selection process for `iterations` number of times...
    while count < iterations:
        # Compute a new q-numerical range value and the according obtained
        # modulus.
        q_range_value, q_radius_value = selection_function(matrix, q_value)

        # Update the q-numerical range and the q-numerical radius.
        q_range.append(q_range_value)
        q_radius = max(q_radius, q_radius_value)

        # Increment the iteration counter.
        count += 1

    # Convert the q-numerical range into a `np.ndarray` object and then convert
    # the complex numbers into pairs of corresponding (x, y) coordinates.
    q_range = np.array(q_range)
    q_range_coordinates = np.array([np.real(q_range), np.imag(q_range)])
    q_range_coordinates = np.transpose(q_range_coordinates)
    # Afterwards, find the convex hull of all the obtained points that surely
    # belong to q-numerical range.
    try:
        q_range_result = ConvexHull(q_range_coordinates)
    except:
        # If there exists no nondegenerate convex hull, then just return the
        # generated points in the form of a `np.ndarray` matrix.
        q_range_result = q_range_coordinates

    return q_range_result, q_radius
