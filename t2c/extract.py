from typing import Optional
import logging

import numpy as np
import fire

from .word_embeddings import load_word_embs
from .estimator import (WordCount,
                        WordEmbeddingSimilarity)
from .utils import (load_dictionary,
                    load_ponizovskiy,
                    normalize_scores,
                    load_tokenizer,
                    load_idf)


logging.basicConfig()
logger = logging.getLogger('Extractor')


def extract(
    input_fn: str,
    out_fn: str,
    word_embs_name_or_path: str,
    alpha: float = 0.5,
    idf_fn: Optional[str] = None,
    is_gensim_model: bool=True,
    is_glove: bool=False,
    binary: bool=False,
    dict_fn: Optional[str]=None,
    tokenizer_fn: Optional[str]=None,
    normalization: Optional[str] = None
) -> None:
    """ Extracting estimated concept relevance scores based on text and concept dictionary

    #. may be 5x ~ 10x faster if everything implemented in numba or cython
    #. some more optimization for current form also should be possible
        1. multi-processing over songs
        2. caching word to word distance (or pre-compute all possible pairs)
        3. better vectorization (or plus GPU computation)

    #. three normalization method considered
        1. None: literally do nothing
        2. 'zscore': standard scaling
        3. 'l2': L-2 normalization. it makes the scores per row unit vector.
        4. 'softmax':
            applying softmax after z-scoring. useful when
            the relative difference of score within document is more important
            then the distribution per value variable. (it makes the per value
            distribution inconsistent, while comparison within document more clear)

    Args:
        input_fn: filename for the input text file. it expects one text per
                  line in the file.
        out_fn: filename for the output score (.csv)
        word_embs_name_or_path:
                the name of the wordembedding or path to the word embedding
                currently it supports most of the model that's supported by
                `gensim-data`_. If it is given as the path to the embeeding,
                it should be either binary/text file compatible with `gensim`,
                or HDF file compatible with
                :obj:`~t2c.word_embeddings.Word2VecLookup` or
                :obj:`gloves.model.GloVe`.
        alpha: weighting factor for the `concept representative term` over
               the other concept terms.
        idf_fn: filename contains the inverse document frequency (IDF) of each
                tokens. If not given, the default IDF is used. The custom IDF
                can be provided as textfile, where each row includes pair of
                token and corresponding IDF, delimited by the tab.
        is_gensim_model: if the embedding is for :obj:`gloves.model.GloVe`,
                         this should be flagged as False.
        is_glove: if the embedding is either for :obj:`gloves.model.GloVe`
                  or text/binary file saved in the `glove format`_,
                  this flag should be set as True.
        binary: if the embedding file is for `gensim` and saved in binary,
                 this flag should be set as True.
        dict_fn: filename to the dictionary. If not given, default dictionary.
                 (personal value dictionary developed by `Ponizovskiy et al.`_)
                 The custom dictionary should be in `json` format such as follows

                 .. code-block:: json

                     {
                        "concept1": ["concept1_term1", "concept1_term2"],
                        "concept2": ["concept2_term1", "concept2_term2"]
                     }

        tokenizer_fn: filename for the pre-trained :obj:`~tokenizers.Tokenizer`.
                      if not given, default tokenizer shipped with the package
                      is loaded.
        normalization:
                selects the normalization method. Refer
                :obj:`~t2c.utils.normalize_scores' for details.

    .. _gensim-data: https://github.com/RaRe-Technologies/gensim-data#available-data
    .. _glove format: https://github.com/stanfordnlp/GloVe
    .. _Ponizovskiy et al.: https://osf.io/vt8nf/?view_only=

    """
    # loading tokenizer
    tokenizer = load_tokenizer(path=tokenizer_fn)
    idf_ = load_idf(idf_fn)

    # convert idf into list of values
    idf = np.zeros((len(idf_),), dtype=np.float64)
    for token, value in idf_.items():
        i = tokenizer.token_to_id(token)
        if i is not None:
            idf[i] = value

    # load ssvs dictionary
    if dict_fn is None:
        terms = load_ponizovskiy()
    else:
        terms = load_dictionary(dict_fn)

    # initialize estimator
    if word_embs_name_or_path == 'wordcount':
        estimator = WordCount(terms, tokenizer)
    else:
        # load word embeddings
        word_embs = load_word_embs(word_embs_name_or_path,
                                   tokenizer, is_gensim_model,
                                   is_glove=is_glove,
                                   binary=binary)

        # loading estimator
        estimator = WordEmbeddingSimilarity(terms,
                                            word_embs,
                                            idf,
                                            alpha=alpha)

    # load input data (we're expecting a text file whose lines are flattened text of docs)
    new_docs = []
    with open(input_fn, 'r') as f:
        for line in f:
            new_docs.append(line.replace('\n', ''))

    # predict and normalize
    S = estimator.predict_scores(new_docs)
    S = normalize_scores(S, normalization)

    # save the data and do not return anything
    S.to_csv(out_fn)


def main():
    fire.Fire(extract)


if __name__ == "__main__":
    main()
