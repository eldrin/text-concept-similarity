from typing import Optional, Union, Literal
from pathlib import Path
from os.path import exists
import logging
import json

import pandas as pd
import numpy as np
import numpy.typing as npt
import requests

from gloves.utils import init_tokenizer
from tokenizers import Tokenizer

from .misc_data import (default_tokenizer,
                        default_idf)


# the default data root is $HOME/.t2c-sim/
DATA_PATH = Path.home().joinpath('.t2c-sim')
if not DATA_PATH.exists():
    DATA_PATH.mkdir(parents=True, exist_ok=True)
PONIZOVSKIY_PATH = DATA_PATH.joinpath('ponizovskiy_pvd.txt')

PONIZOVSKIY_VALUE_MAP = {
    '1': 'security',       # 'SE'
    '2': 'conformity',     # 'CO'
    '3': 'tradition',      # 'TR'
    '4': 'benevolence',    # 'BE'
    '5': 'universalism',   # 'UN'
    '6': 'self-direction', # 'SD'
    '7': 'stimulation',    # 'ST'
    '8': 'hedonism',       # 'HE'
    '9': 'achievement',    # 'AC'
    '10': 'power'          # 'PO'
}


logger = logging.getLogger('utils')


def normalize_scores(
    scores: Union[list[dict[str, float]],
                  list[dict[str, npt.NDArray[np.float64]]]],
    method: Literal[None, 'zscore', 'l2', 'softmax']
) -> pd.DataFrame:
    """ normalize the socres with different strategies

    it normalizes the estimated scores with different strategies.
    Currently, 4 strategies (including null) are supported:

    1. null (None)
        :no normalization. just wrapping the given input to pandas.DataFrame
         and return.

    2. z-scoring ('zscore')
        :"standardize" the the scores per concepts with z-scoring

    3. l2-normalization ('l2')
        :normalize the scores per "instance/row" (not per concept), using
         l2-norm of each row. Each row becomes the unit vectors.

    4. softmax ('softmax')
        :applying the softmax per "instance/row". It transforms the real
         floating values into simplex (that sums to one.)

    Note:
        z-scoring is conducted as 'preprocessing' for 'l2' and 'softmax'
        normalization.o

    Args:
        scores: scores to be normalizes.
        method: normalization method indicator.

    Returns:
        normalized score.
    """
    # input check
    scores_ = pd.DataFrame(scores)

    if method is None:
        return scores_

    # standardization is done in both ways
    z = (scores_ - scores_.mean(0)) / scores_.std(0)

    if method == 'zscore':
        return z

    if method == 'l2':
        return z / np.linalg.norm(z.values, axis=1)[:, None]

    if method == 'softmax':
        y = np.exp(z)
        return y / y.values.sum(1)[:, None]

    raise ValueError(
        '[ERROR] method should be one of '
        '{None, "zscore", "l2", "softmax"}!'
    )


def load_idf(
    path: Optional[str] = None
) -> dict[str, float]:
    """ loads IDF values

    it loads the IDF values. if no specific filename is given, the default IDF
    values from the package is loaded instaed.

    A custom IDF can be given. A text file where each row has the term - idf
    pair delimited by the tabbed space can be parsed and used for the further
    procedure.

    .. code-block:: text

        apple\t11.1251
        pear\t12.1205
        ...

    Args:
        path: filename to the IDF values for terms.

    Returns:
        dictionary contains IDF values per term.
    """
    if path is None:
        path = default_idf()
    else:
        if not exists(path):
            logger.warning('Could not find idf file! '
                           'falling back to the default idf...')
            path = default_idf()

    idf_dict = {}
    with open(path, 'r') as f:
        for line in f:
            token, idf_ = line.replace('\n', '').split('\t')
            idf_dict[token] = float(idf_)
    return idf_dict


def load_tokenizer(path: Optional[str]=None) -> Tokenizer:
    """ loads pretrained tokenizer.

    it loads the pretrained tokenizer configurations. if no specific filename
    is given, the default tokenizer is loaded instead.

    Args:
        path: filename to the pretrained tokenizer dump file.

    Returns:
        :obj:`tokenizers.Tokenizer` loaded with pretrained configuration.
    """
    if path is None:
        logger.warning('tokenizer is not given. falling back to the default '
                       'tokenizer from the package...')
        path = default_tokenizer()
    else:
        if not exists(path):
            logger.warning('Could not find tokenizer dump file! '
                           'falling back to the default tokenizer...')
            path = default_tokenizer()

    tokenizer = init_tokenizer(path=path)
    return tokenizer


def load_dictionary(
    path: str
) -> dict[str, set[str]]:
    """ load dictionary file.

    It just reads the custom dictionary file in the json format.

    Args:
        path: filename to the custom dictionary filename

    Returns:
        dictionary of concept-words.
    """
    with Path(path).open('r') as fp:
        dic = {k:set(v) for k, v in json.load(fp).items()}
    return dic


def _fetch(
    url: str,
    path: Path,
    chunk_size: int=128
) -> str:
    """ fetch files from internet.

    a simple routine to fetch the small file from the url

    Args:
        url: target url
        path: output directory
        chunk_size: integer chunk size of the binary that is going to be fetched

    Returns:
        filename where the file is downloaded.
    """
    # check if it's already there
    if path.exists():
        return path.absolute().as_posix()

    # otherwise, fetch it
    logger.info(f'{path.as_posix()} not found. try fetching...')
    r = requests.get(url, stream=True)
    with path.open('wb') as f:
        for chunk in r.iter_content(chunk_size):
            f.write(chunk)
    logger.info(f'{path.as_posix()} downloaded!')
    return path.absolute().as_posix()


def _fetch_ponizovskiy(
    url: str = 'https://osf.io/vy475/download'
) -> None:
    """ Fetching Personal Values Dictionary by Ponizovskiy et al.

    it fetches the personal value dictionary developed by Ponizovskiy et al. (2020)

    Args:
        url: url of the personal value dictionary from Ponizovskiy et al.
             the default is set as their OSF entry.
    """
    _fetch(url, PONIZOVSKIY_PATH)


def load_ponizovskiy() -> dict[str, set[str]]:
    """ load personal value dictionary from Ponizovskiy et al.

    it loads the personal value dictionary from Ponizovskiy et al. It is used
    as the default dictionary if no custom dictionary is given.

    Returns:
        the personal value dictionary developed by Ponizovskiy et al. (2020)
    """
    body_starts_at = 12

    if not PONIZOVSKIY_PATH.exists():
        _fetch_ponizovskiy()

    # initualize the output
    value_terms = {val:set() for val in PONIZOVSKIY_VALUE_MAP.values()}
    with PONIZOVSKIY_PATH.open() as f:
        for i, line in enumerate(f):
            # register body
            if i >= body_starts_at:
                parsed = line.replace('\n', '').split('\t')
                if parsed[1] == '':
                    continue
                term, key = parsed
                value_terms[PONIZOVSKIY_VALUE_MAP[key]].add(term)

    return value_terms
