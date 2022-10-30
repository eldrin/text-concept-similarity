from pathlib import Path
import sys
sys.path.append(Path(__file__).parent.parent.as_posix())

from itertools import chain

import numpy as np
import pandas as pd
from t2c.estimator import WordCount
from t2c.utils import load_tokenizer


def test_wordcount():

    dic = {
        'a': {'1', '2', '3'},
        'b': {'4', '5', '6'},
        'c': {'7', '8'}
    }
    # total_terms = [4, 4, 3, 1]
    total_terms = set(chain.from_iterable(dic.values()))

    inp = [
        '1 1 4 7',
        '4 5 5 8',
        '1 4 8',
        '1 9'  # contains the term that's not in the dictionary
    ]
    n_concept_terms = [
        len(list(filter(lambda x: x in total_terms, i.split(' '))))
        for i in inp
    ]

    expected_without_ipsatization = [
        {'a': 2, 'b': 1, 'c': 1},
        {'a': 0, 'b': 3, 'c': 1},
        {'a': 1, 'b': 1, 'c': 1},
        {'a': 1, 'b': 0, 'c': 0}
    ]

    # expected_with_ipsatization = [
    #     {'a': -2, 'b': -3, 'c': -3},
    #     {'a': -4, 'b': -1, 'c': -3},
    #     {'a': -2, 'b': -2, 'c': -2},
    #     {'a': 0,  'b': -1, 'c': -1}
    # ]
    expected_with_ipsatization = [
        {k: v - tot for k, v in row.items()}
        for row, tot
        in zip(expected_without_ipsatization, n_concept_terms)
    ]

    tok = load_tokenizer()

    # 1 - no ipsatize
    wc = WordCount(dic, tok, ipsatize=False)
    res = wc.predict_scores(inp)
    assert np.equal(pd.DataFrame(res).values,
                    pd.DataFrame(expected_without_ipsatization).values).all()


    # 2 - ipsatization
    wc = WordCount(dic, tok, ipsatize=True)
    res = wc.predict_scores(inp)
    assert np.equal(pd.DataFrame(res).values,
                    pd.DataFrame(expected_with_ipsatization).values).all()
