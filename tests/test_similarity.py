from pathlib import Path
import sys
sys.path.append(Path(__file__).parent.parent.as_posix())

import numpy as np
import pandas as pd
from t2c.estimator import (normalize_vector,
                           weight_vector,
                           cosine_similarity_with_unit_vectors)
from t2c.utils import normalize_scores


def test_normalize_vector():

    # 1 - normal case
    inp = np.array([[0, 1, 0, 1, 1]])
    norm = 1.7320508075688772
    expected = inp / norm

    res = normalize_vector(inp)
    assert np.isclose(res, expected).all()

    # 2 - zero entry
    inp = np.array([[0, 0, 0, 0]])
    norm = 0
    expected = inp

    res = normalize_vector(inp)
    assert np.isclose(res, expected).all()

    # 3 - including NaN
    inp = np.array([[np.nan, 0, 1]])

    # TODO: should it raise error?
    # (and this is not used, just for readability)
    expected = np.array([[np.nan, np.nan, np.nan]])

    res = normalize_vector(inp)
    assert np.isnan(res).all()

    # TODO: add shape related tests later


def test_weight_vector():

    vectors = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    # 1 - normal case, but scalar
    weight = 1.

    res = weight_vector(vectors, weight)
    assert np.equal(vectors, res).all()

    # 2 - normal case, scalar but in 1d vector w/ 1d array
    weight = np.array([1.])

    res = weight_vector(vectors, weight)
    assert np.equal(vectors, res).all()

    # 3 - normal case, vector
    weight = np.array([1., 0., 1.])
    expected = np.array([
        [0, 1, 1],
        [0, 0, 0],
        [1, 1, 1]
    ])

    res = weight_vector(vectors, weight)
    assert np.equal(res, expected).all()

    # 4 - vector, shape mismatch
    weight = np.array([1., 1.])
    try:
        res = weight_vector(vectors, weight)
    except ValueError:
        pass
    else:
        raise ValueError('[ERROR] mismatch should raise error!')

    # 5 - normal case, matrix
    weight = np.array([
        [1., 1., 1.],
        [0., 0., 0.],
        [0., 0., 1.]
    ])
    expected = np.array([
        [0., 1., 1.],
        [0., 0., 0.],
        [0., 0., 1.]
    ])
    res = weight_vector(vectors, weight)
    assert np.equal(res, expected).all()

    # 6 - matrix, shape mismatch
    weight = np.array([
        [1., 1.],
        [0., 1.]
    ])
    try:
        res = weight_vector(vectors, weight)
    except ValueError:
        pass
    else:
        raise ValueError('[ERROR] mismatch should raise error!')


def test_cosine_similarity_with_unit_vectors():
    """
    with pre-normalized condition, it's really a simple dot product.
    """

    A = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    B = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1]
    ])

    expected = np.array([
        [1., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 1.]
    ])

    res = cosine_similarity_with_unit_vectors(A, B)
    assert np.equal(expected, res).all()


def test_normalize_score():

    inp = [
        {'c1': 1., 'c2': 1.},
        {'c1': 0., 'c2': 1.},
        {'c1': 1., 'c2': 0.},
    ]

    # 1 - null
    res = normalize_scores(inp, None)
    assert np.equal(pd.DataFrame(inp).values, res.values).all()

    # 2 - zscore
    res = normalize_scores(inp, 'zscore')
    expected = np.array([
        [ 0.577350,  0.577350],
        [-1.154701,  0.577350],
        [ 0.577350, -1.154701]
    ])
    assert np.isclose(res.values, expected).all()

    # 3 - l2 normalization (after zscore)
    res = normalize_scores(inp, 'l2')
    expected = np.array([
       [ 0.70710678,  0.70710678],
       [-0.89442719,  0.4472136 ],
       [ 0.4472136 , -0.89442719]
    ])
    assert np.isclose(res.values, expected).all()

    # 4 - softmax (after zscore)
    res = normalize_scores(inp, 'softmax')
    expected = np.array([
       [0.5       , 0.5       ],
       [0.15032545, 0.84967455],
       [0.84967455, 0.15032545]
    ])
    assert np.isclose(res.values, expected).all()

    # TODO: case where irregularities contained
    # (i.e., all zero, all same values, including nan, etc.)
