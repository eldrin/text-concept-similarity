from typing import Union, Any
from collections import Counter
import logging
from itertools import chain

import numpy as np
import numpy.typing as npt

from tokenizers import Tokenizer
from gloves.model import GloVe
from .word_embeddings import GensimWordEmbedding


logger = logging.getLogger('text2conceptsim')


def normalize_vector(
    vector: npt.NDArray[np.float64],
    eps: float = 1e-12
) -> npt.NDArray[np.float64]:
    """ normalize the vectors into the unit vector

    it devide given vectors with their L2 norm, so that the outcome
    is to be the unit vector of them.

    Args:
        vector: set of vectors to be normalized. expecting first dimension to be
                the number of vectors, second being the dimensionality
        eps: small floating number that prevents the 0 vector to be explode

    Returns:
        normalized unit vectors of given vectors.
    """
    # then normalize them for cosine distance
    denum = np.maximum(np.linalg.norm(vector, axis=1)[:, None], eps)
    vector = (vector / denum)
    return vector


def weight_vector(
    vector: npt.NDArray[np.float64],
    weight: Union[float, npt.NDArray[np.float64]]
) -> npt.NDArray[np.float64]:
    """ weight the vectors with given weight values.

    It multiplies the weight values to each value. If first dimension of the weight
     is given differently to the vectors, it broadcast.

    Args:
        vector: set of vectors to be weighted. expecting first dimension to be
                the number of vectors, second being the dimensionality
        weight: tensor (or scalar) contains weight value to be applied.

    Returns:
        weighted vectors. It has same shape of input vector
    """
    # wrap weight
    weight = np.array(weight)

    # check shapes: we expect the row vectors (#vectors, dimensionality)
    n_vectors, _ = vector.shape

    if weight.shape == vector.shape:
        # the weight is somehow expanded
        # (either per dimension weight or broadcastable already)
        out = vector * weight

    elif weight.ndim == 0:
        # scalar without dimensionality
        out = vector * weight

    elif weight.ndim == 1:

        # the weight is given as either scalar or 1d vector
        if len(weight) == 1:
            # scalar, but in 1 dimensional vector w/ 1 elem
            out = vector * weight
        else:
            if len(weight) == n_vectors:
                out = vector * weight[:, None]  # we're happy
            else:
                # this is not good
                raise ValueError(
                    '[ERROR] first dimension (length) of weight '
                    'and vector should be mathced!'
                )
    else:
        # this is not good
        raise ValueError(
            '[ERROR] first dimension (length) of weight '
            'and vector should be mathced!'
        )

    return out


def cosine_similarity_with_unit_vectors(
    A: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """ compute cosine similarity between bag of unit vectors

    as we assume the vectors are already normalized, essentially
    it just computes the dot-products between to matrices.

    it expect the size of:
        A (left vectors) having (n_vectors_A, dim)
        B (right vectors) having (n_vectors_B, dim)

    Thus the output size will be (n_vectors_A, n_vectors_B)

    Args:
        A: left unit vectors
        B: right unit vectors

    Returns:
        computed cosine distances between two sets of vectors
    """
    # TODO: should we test the "unit vector"ness here? <- probably yes. but later.
    return A @ B.T


class BaseText2ConceptEstimator(object):
    """
    """
    def __init__(
        self,
        dictionary: dict[str, set[str]],
        *args,
        **kwargs
    ):
        """
        """
        self.dictionary = dictionary
        self.init_dictionary(dictionary)

    def init_dictionary(
        self,
        dictionary: dict[str, set[str]]
    ):
        """
        """
        raise NotImplementedError()

    def predict_scores(
        self,
        new_doc_batch: list[str],
    ) -> list[dict[Any, float]]:
        """
        """
        raise NotImplementedError()


class WordCount(BaseText2ConceptEstimator):
    """
    TODO: maybe stemming and other pre-processing stuffs for improving hit-rate
    """
    def __init__(
        self,
        dictionary: dict[str, set[str]],
        tokenizer: Tokenizer,
        *args,
        **kwargs
    ):
        """
        """
        super().__init__(dictionary)
        self.tokenizer = tokenizer

    def init_dictionary(
        self,
        dictionary: dict[str, list[str]]
    ):
        """
        """
        self.dictionary_inv = dict(chain.from_iterable([
            [(vv, k) for vv in v] for k, v in dictionary.items()
        ]))

    def predict_scores(
        self,
        new_doc_batch: list[str],
    ) -> list[dict[Any, float]]:
        """
        """
        value_scores = []
        for doc in new_doc_batch:
            # encode the string into tokens using tokenizer
            tok = self.tokenizer.encode(doc)

            # get term-frequency of them
            tf = Counter(tok.tokens)

            doc_value_sim = {k:0 for k in self.dictionary.keys()}
            for word, count in tf.items():
                concept = self.dictionary_inv.get(word)
                if concept is not None:
                    doc_value_sim[concept] += count
            value_scores.append(doc_value_sim)
        return value_scores


class WordEmbeddingSimilarity(BaseText2ConceptEstimator):
    """
    """
    def __init__(
        self,
        dictionary: dict[str, set[str]],
        word_embs: Union[GensimWordEmbedding, GloVe],
        idf: npt.NDArray[np.float64],
        *args,
        **kwargs
    ):
        """
        idf is list of float containing teh inverse-document frequency
        and it's indices are synced with the tokenizer

        for now, we only apply idf to text tokens, as dictionary already implicitly
        has prior that concept-terms are equally important.

        TODO: We probably should revisit here later to generalize
              to apply concept-term weights as some study does
              explore and ship the concept-term weight as part of the dictionary.
        TODO: combining [word_embedding - tokenzier - idf] might be a good idea
              to improve the usability
        """
        self.word_embs = word_embs
        self.idf = idf  # list of float that sorted with given tokenizer
        super().__init__(dictionary)

    def init_dictionary(
        self,
        dictionary: dict[str, set[str]]
    ):
        """
        """
        # init value embeddings for further computations
        self.concept_embs = dict()
        for concept, terms in dictionary.items():
            # fetch embeddings and post processing
            terms_embs, terms_idfs, terms_healthy = self._get_embs_idf(list(terms))
            # terms_embs = weight_vector(normalize_vector(terms_embs), terms_idfs)
            cncpt_emb, cncpt_idf, cncpt_healthy = self._get_embs_idf(concept)

            # # if there IS embedding, but there's no idf, we assign median IDF
            # if cncpt_emb.shape[0] > 0 and len(cncpt_idf) == 0:
            #     cncpt_idf = np.array([np.median(self.idf[np.where(self.idf != 0)])])

            # cncpt_emb = weight_vector(normalize_vector(cncpt_emb), cncpt_idf)

            # get relative contributions
            weight = (
                len(cncpt_healthy)
                / (len(cncpt_healthy) + (len(terms_healthy) / len(terms)))
            )

            # get the final embedding slist
            embs = np.concatenate(
                [
                    (terms_embs / terms_embs.shape[0]) * (1 - weight),
                    cncpt_emb * weight
                ],
                axis=0
            )
            self.concept_embs[concept] = embs

    def __fetch_idfs(
        self,
        token_ids: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """
        """
        return self.idf[token_ids]

    def __get_healthy_terms(
        self,
        terms: Union[str, list[str]],
    ) -> tuple[npt.NDArray[np.int64],
               npt.NDArray[np.int64],
               list[Union[str, bytes]]]:
        """
        """
        # get terms and tokens that are "healthy"
        terms_ary = np.asarray(terms)

        token_ids = np.array(
            [self.word_embs._tokenizer.token_to_id(w) for w in terms]
        )
        healthy = np.where(token_ids != None)[0]

        # filter unhealthy entries
        return (
            healthy,
            token_ids[healthy].astype(np.int64),
            terms_ary[healthy].tolist()
        )

    def __get_embs(
        self,
        terms_ary: list[Union[str, bytes]]
    ) -> npt.NDArray[np.float64]:
        """
        """
        # get embeddings
        if len(terms_ary) > 0:
            embs = np.array(self.word_embs[terms_ary])
        else:
            embs = np.random.randn(1, self.word_embs.n_components)

        # fill empty rows with random values
        where_nans = np.sum(embs, axis=1) == 0
        embs[where_nans] = np.random.randn(where_nans.sum(), embs.shape[-1])

        return embs

    def _get_embs_idf(
        self,
        terms: Union[str, list[str]]
    ) -> tuple[npt.NDArray[np.float64],
               npt.NDArray[np.float64],
               npt.NDArray[np.int64]]:
        """
        """
        if isinstance(terms, str):
            return self._get_embs_idf([terms])

        # get terms and tokens that are "healthy"
        healthy, token_ids, terms_ary = self.__get_healthy_terms(terms)

        # get idfs
        idfs = self.__fetch_idfs(token_ids)

        # get embs
        embs = self.__get_embs(terms_ary)

        return embs, idfs, healthy

    def predict_scores(
        self,
        new_doc_batch: list[str],
    ) -> list[dict[str, npt.NDArray[np.float64]]]:
        """ compute weighted average score between document and values

        s_{value, doc} = sum_{i, j} cosim(val_w_i, w_j) * idf(val_w_i) * tfidf(w_j)

        i \in values, j \in doc

        NOTE: for performance, we pre-normalize all vectors and wegithed with
              corresponding (tf)idf, and it just computed by the inner product
              between pre-weighted value embeddings and term embeddings (in input document)

        this function can be faster with either
        numba or cython implementation (~5x or 10x easily)
        """
        value_scores = []
        for doc in new_doc_batch:

            # encode the string into tokens using tokenizer
            tok = self.word_embs._tokenizer.encode(doc)

            # get term-frequency of them
            tf = Counter(tok.tokens)

            tokens, counts = tuple(map(np.array, zip(*tf.items())))
            embs, idfs, healthy = self._get_embs_idf(list(tokens))

            if embs.size == 0:
                # no words are recognized by word2vec:
                #    draw normal random vector and normalize to be a unit vector
                logger.warning(
                    'No words are recognized by word embedding! '
                    'Drawing a random vector instead...'
                )
                weighted_embs = np.random.randn(1, self.word_embs.n_components)
                weighted_embs /= np.linalg.norm(weighted_embs)

            else:
                # get the IDF weighted unit vectors
                weighted_embs = weight_vector(normalize_vector(embs), idfs)

                # then again get the counts weighted (and normalized) vectors
                counts = np.array(counts)[healthy]
                weighted_embs = weight_vector(weighted_embs, counts)
                weighted_embs /= counts.sum()

            # compute per value weighted average cosine distances
            doc_concept_sim = {}
            for concept, concept_w_embs in self.concept_embs.items():
                doc_concept_sim[concept] = (
                    cosine_similarity_with_unit_vectors(concept_w_embs,
                                                        weighted_embs).sum()
                )
            value_scores.append(doc_concept_sim)
        return value_scores
