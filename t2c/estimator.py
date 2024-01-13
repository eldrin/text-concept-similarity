from typing import Union, Any, Optional
from collections import Counter
import logging
from itertools import chain

import numpy as np
import numpy.typing as npt

from tokenizers import Tokenizer
from gloves.model import GloVe
from .word_embeddings import WordEmbedding


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

    it multiplies the weight values to each value. If first dimension of the weight
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
    It is a abstract class for the estimator. It defines a few methods and
    fields that are necessary for the estimatino.

    Attributes:
        dictionary (dict[str, set[str]]):
            a dictionary whose key is the concept word, value contains a
            set of words that are related to the concept.
    """
    def __init__(
        self,
        dictionary: dict[str, set[str]],
        *args,
        **kwargs
    ):
        self.dictionary = dictionary
        self.init_dictionary(dictionary)

    def init_dictionary(
        self,
        dictionary: dict[str, set[str]]
    ):
        """ initialize the dictionary

        Initializes the dictionary suitable for each esimtation method.

        Args:
            dictionary: a dictionary whose key is the concept word, value
                        contains a set of words that are related to the concept.
        """
        raise NotImplementedError()

    def predict_scores(
        self,
        new_doc_batch: list[str],
    ) -> list[dict[str, float]]:
        """ predict estimated simliarity between text and concepts

        similarity esimtation between text and each concept is computed.

        Args:
            new_doc_batch: given new batch of texts.

        Returns:
            esimated simliarity scores per concpet, per text.
        """
        raise NotImplementedError()


class WordCount(BaseText2ConceptEstimator):
    """
    It implements simple word-counting based estimation. We follow the procdure
    that is used by (Ponizovskiy et al. 2020).

    For given document :math:`d`, it compute the scores per concept
    :math:`y_{c, d}` based on the frequency of words per concept **minus**
    the total hit count from all words within the dictionary.

    .. math::
        y_{c, d} &= \\text{tf}_{c, d} - \\text{tf}_{d} \\\\
        \\text{tf}_{c, d} &= |t \\in \\mathcal{T}_{c} : t \\in d| \\\\
        \\text{tf}_{d} &= \\sum_{c\\in\\mathcal{C}} \\text{tf}_{c, d}

    where :math:`c\\in\\mathcal{C}` denotes a concept, :math:`d\\in\\mathcal{D}`
    denotes a document within the corpus :math:`\\mathcal{D}`, and
    :math:`t\\in d` denotes the term/word/token within the document.

    The (document, concept) specific term frequency :math:`tf` is "ipsatized"
    following (Ponizovskiy et al. 2020), which works as a sort of normalization
    that transforms the linear scores become more aligned with "ranking" of
    concepts. Specifically, it is computed by subtracting the frequency of
    all terms included in the dictionary from the frequencies of each concept.
    (i.e., the first equation above.)

    Example::

        from t2c.estimator import WordCount
        from t2c.utils import (load_ponizovskiy,
                               load_tokenizer)

        # load default personal value dictionary
        dic = load_ponizovskiy()

        # load default tokenizer
        tok = load_tokenizer()

        inp = [
            "There is nothing either good or bad, but thinking makes it so.",
            ...
        ]

        # instantiate
        est = WordCount(dic, tok, ipsatize=True)

        # predict concept scores
        scores = est.predict_scores(inp)


    Attributes:
        dictionary (dict[str, set[str]]):
            a dictionary whose key is the concept word, value contains a
            set of words that are related to the concept.
        tokenizer (:obj:`tokenizers.Toeknizer`):
            a pretrained tokenizer which do the toknization of given text.
        ipsatize (bool): flag that sets whether "ipsatization" is applied or not.
    """
    def __init__(
        self,
        dictionary: dict[str, set[str]],
        tokenizer: Tokenizer,
        ipsatize: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(dictionary)
        self.tokenizer = tokenizer
        self.ipsatize = ipsatize

    def init_dictionary(
        self,
        dictionary: dict[str, list[str]]
    ):
        """ initialize the dictionary

        The dictionary is reversed so that one can efficiently check the
        hit of each tokens to the set of tokens belonging a certain concept.

        Args:
            dictionary: a dictionary whose key is the concept word, value
                        contains a set of words that are related to the concept.
        """
        self.dictionary_inv = dict(chain.from_iterable([
            [(vv, k) for vv in v] for k, v in dictionary.items()
        ]))

    def _preproc(self, word: str) -> str:
        """ preprocessing of given word/token

        In particular, we strip the place holder `Ġ` that represents the
        whitespace used in the :obj:`tokenizers.Tokenizer`.

        Args:
            word: a token that is to be pre-processed
        """
        return word.replace('Ġ', '').replace(' ', '')

    def predict_scores(
        self,
        new_doc_batch: list[str],
    ) -> list[dict[Any, float]]:
        """ predict estimated simliarity between text and concepts

        similarity esimtation between text and each concept is computed.

        Args:
            new_doc_batch: given new batch of texts.

        Returns:
            esimated simliarity scores per concpet, per text.
        """
        value_scores = []
        for doc in new_doc_batch:
            # encode the string into tokens using tokenizer
            tok = self.tokenizer.encode(doc)

            # get term-frequency of them
            tf = Counter(tok.tokens)

            doc_value_sim = {k:0 for k in self.dictionary.keys()}
            total_count = 0
            for word, count in tf.items():
                concept = self.dictionary_inv.get(self._preproc(word))
                if concept is not None:
                    doc_value_sim[concept] += count
                    total_count += count

            if self.ipsatize:
                for concept in doc_value_sim.keys():
                    doc_value_sim[concept] -= total_count

            value_scores.append(doc_value_sim)
        return value_scores


class WordEmbeddingSimilarity(BaseText2ConceptEstimator):
    """
    It implements the text to concept similarity via the cosine similarity
    between word-embeddings. Additionally, it weighs each word/token within
    a document by the `inverse document frequency`_ (IDF). It weighs less
    the overly frequent words (i.e., such as stopwords) while weigh more the
    other words.

    For a given concept :math:`c\\in\\mathcal{C}` and document
    :math:`d\\in\\mathcal{D}`, the score :math:`y_{c, d}` is computed as follows:

    .. math::
       :nowrap:

       \\[
       y_{c, d} =
       \\begin{cases}
         \\tilde{y}_{\\{{t^\\prime}_{c}\\}, d}, & \\text{if } |\\mathcal{T}_{c}\\setminus {t^\\prime}_{c}| = 0 \\wedge {t^\\prime}_{c} \\in \\mathcal{W} \\\\
         \\tilde{y}_{\\mathcal{T}_{c}\\setminus {t^\\prime}_{c}, d},     & \\text{if } |\\mathcal{T}_{c}\\setminus {t^\\prime}_{c}| \\gt 0 \\wedge {t^\\prime}_{c} \\notin \\mathcal{W} \\\\
         0,                                        & \\text{if } |\\mathcal{T}_{c}\\setminus {t^\\prime}_{c}| = 0 \\wedge {t^\\prime}_{c} \\notin \\mathcal{W} \\\\
         \\alpha\\tilde{y}_{\\{{t^\\prime}_{c}\\}, d} + (1 - \\alpha)\\tilde{y}_{\\mathcal{T}_{c}\\setminus {t^\\prime}_{c}, d} & \\text{otherwise}
       \\end{cases}
       \\]

    where :math:`\\mathcal{T}_{c} := \\{ t \\in c : t \\in \\mathcal{W} \\}`
    denotes the set of terms/words/tokens related to a concept :math:`c`,
    :math:`t^{\\prime}_{c}` refers the "representative term" of the concept
    :math:`c` (i.e., *openness* vs. {new, imaginative, untraditional, ...}),
    :math:`\\mathcal{W}` means the set of words that are supported by the embedding
    :math:`\\mathcal{E}_{\\mathcal{W}} := \\{ e_{t} \\in \\mathcal{E} : t \\in \\mathcal{W}\\ \\wedge \\mathcal{E} \\subset \\mathbb{R}^{r} \\}`.

    :math:`\\alpha\\in[0, 1]` controls the contribution of representative
    term :math:`t^{\\prime}_{c}` over the other related terms in
    :math:`\\mathcal{T}_{c}`. If :math:`\\alpha` is set as 1, it means that
    it does not consider the sub scores from other terms, and vice versa
    when it is set as 0. Further, the score can visit corner cases where either
    representative term :math:`t^{\\prime}_{c}` or all of other concept
    terms :math:`t_{c}\\in\\mathcal{T}_{c}` are not found from the embedding
    :math:`\\mathcal{E}_{\\mathcal{W}}`. In those cases, first three conditional
    score is computed and returned.

    Finally, the intermediate score :math:`\\tilde{y}_{\\mathcal{T}, d}` is computed as follows.

    .. math::
        \\tilde{y}_{\\mathcal{T}, d} = \\frac{ \\sum_{k \\in \\mathcal{T}}\\sum_{t \\in d} s_{\\text{cos}}(k, t)\\text{tf}(t)\\text{idf}(t) }{ |\\mathcal{T}| \\sum_{t \\in d} \\text{tf}(t) }

    where :math:`\\mathcal{T}` denotes the set of terms/words/tokens,
    :math:`s_{\\text{cos}}` refers the cosine simliarity between word embeddings
    corresponding terms :math:`k` and :math:`t`.

    .. math::
        s_{\\text{cos}}(a, b) = \\frac{ e_{a} \\cdot e_{b} }{ ||e_{a}||||e_{b}|| }

    where :math:`e_{t}\\in\\mathcal{E}_{\\mathcal{W}}` denotes the embedding vector
    with the dimensionality of :math:`r`, and :math:`||\\cdot||` refers the
    L-2 norm of the vector.

    In this procedure, :math:`\\text{idf}` is computed as follows:

    .. math::
        \\text{idf}(t) = \\text{log}(|\\mathcal{D}| / (1 + \\text{df}(t))) + 1

    where :math:`\\text{df}(t)` denotes the *document frequency* of
    term/word/token :math:`t`. IDF is pre-computed from a large corpus such
    as the `Wikipedia`_ (which is our default). Custom IDF can be given, if
    it's following the format. (i.e., text file where each row contains term
    and IDF value delimited by tab)

    Example::

        from t2c.estimator import WordEmbeddingSimilarity
        from t2c.word_embeddings import load_word_embs
        from t2c.utils import (load_ponizovskiy,
                               load_tokenizer,
                               load_idf)

        # load default personal value dictionary
        dic = load_ponizovskiy()

        # load default tokenizer
        tok = load_tokenizer()

        # load default idfs
        idf = load_idf()

        # load word embedding
        word_embs = load_word_embs(word_embs_name_or_path,
                                   tok)

        inp = [
            "There is nothing either good or bad, but thinking makes it so.",
            ...
        ]

        # instantiate
        est = WordEmbeddingSimilarity(dic, word_embs, idf,
                                      alpha=0.5)

        # predict concept scores
        scores = est.predict_scores(inp)


    TODO: We probably should revisit here later to generalize
          to apply concept-term weights as some study does
          explore and ship the concept-term weight as part of the dictionary.
    TODO: combining [word_embedding - tokenzier - idf] might be a good idea
          to improve the usability

    Note:
        In case the term/token doesn't have the IDF entry,
        we assign the median of IDF.

    Attributes:
        dictionary (dict[str, set[str]]):
            a dictionary whose key is the concept word, value contains a
            set of words that are related to the concept.
        word_embs (:obj:`~t2c.word_embeddings.WordEmbedding`):
            word embedding object. see :obj:`~t2c.word_embeddings.WordEmbedding`.
        idf (:obj:`numpy.typing.NDArray`, optional):
            a float array contains IDF values per token/term/words. Its indices
            is pre-sorted based on the tokenizer embedded in the `word_embs`.
            if not given, it weights the scores per terms uniformly.
            (the weight is set to 1)
        alpha (float):
            a float ranged within [0, 1] that controls the relative importance of
            `representative term` when computing the aggregated score

    .. _inverse document frequency: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    .. _Wikipedia: https://www.wikipedia.org
    """
    def __init__(
        self,
        dictionary: dict[str, set[str]],
        word_embs: WordEmbedding,
        alpha: float = 0.5,
        idf: Optional[dict[str, float]] = None,
        *args,
        **kwargs
    ):
        self.word_embs = word_embs
        self.idf = idf  # list of float that sorted with given tokenizer
        self.alpha = alpha
        super().__init__(dictionary)

    def init_dictionary(
        self,
        dictionary: dict[str, set[str]]
    ):
        """ initialize the dictionary

        Args:
            dictionary: a dictionary whose key is the concept word, value
                        contains a set of words that are related to the concept.
        """
        # init value embeddings for further computations
        self.concept_embs = dict()
        for concept, terms in dictionary.items():
            # fetch embeddings and post processing
            if isinstance(self.word_embs.w2v, GloVe):
                concept_terms = f"{concept} {' '.join(terms)}"
                concept_terms = (
                    self.word_embs.w2v._tokenizer.encode(concept_terms).tokens
                )
                embs, _, healthy = self._get_embs_idf(concept_terms)
                cncpt_emb, terms_embs = embs[0][None], embs[1:]
                cncpt_healthy, terms_healthy = np.array([healthy[0]]), healthy[1:]
            else:
                terms_embs, _, terms_healthy = self._get_embs_idf(list(terms))
                cncpt_emb, _, cncpt_healthy = self._get_embs_idf(concept)

            # get relative contributions
            if len(terms_healthy) == 0 and len(cncpt_healthy) == 1:
                weight = 1.
            elif len(terms_healthy) > 0 and len(cncpt_healthy) == 0:
                weight = 0.
            elif len(terms_healthy) == 0 and len(cncpt_healthy) == 0:
                weight = None
            else:
                weight = self.alpha

            # get the final embedding slist
            if weight is not None:
                a = weight
                b = 1 - weight
            else:
                a = 0.
                b = 0.

            embs = np.concatenate(
                [
                    (terms_embs / terms_embs.shape[0]) * b,
                    cncpt_emb * a
                ],
                axis=0
            )
            self.concept_embs[concept] = embs

    def __fetch_idfs(
        self,
        tokens: npt.NDArray[object]
    ) -> npt.NDArray[np.float64]:
        """ fetch IDF values from internal idf vector

        it takes "indices" of the term through the tokenizer shipped with
        the embedding (word_embs member of the object).

        Args:
            tokens: an array of indices corresponding to the terms of which
                       the IDFs to be retrieved.

        Returns:
            IDF values corresponding to the input indices.
        """
        return (
            np.ones(len(tokens))
            if self.idf is None
            else np.array([self.idf.get(t, 0.) for t in tokens])
        )

    def __get_healthy_terms(
        self,
        terms: Union[str, list[str]],
    ) -> tuple[npt.NDArray[np.int64],
               npt.NDArray[np.int64],
               list[Union[str, bytes]]]:
        """ determine and return the healthiness and token ids

        it first determines whether the tokens exist in the embedding.
        if it does, it returns the corresponding token ids (int) and
        its existence in the embedding (int indices of tokens in the tokenizer).

        As it can take both single term (str) or list of terms, it returns
        :obj:`numpy.ndarray` for both cases for generality.

        Finally, it also returns the terms themselves wrapped by the
        :obj:`numpy.ndarray`.

        Args:
            terms: input single or list of terms

        Returns:
            it returns the tuple of: 1) array of indices of terms that exist,
            2) array of tokens ids that exist, and 3) array-wrapped terms
        """
        # get terms and tokens that are "healthy"
        terms_ary = np.asarray(terms)

        token_ids = np.array([self.word_embs.get_id(w) for w in terms])
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
        """ fetch embeddings from the array of terms

        it draws the embeddings corresponding to the list of terms.

        Args:
            terms: (list of) term(s) whose embedding is to be drawn

        Returns:
            embedding vectors corresponding to the terms.
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
        """ get embeddings and corresponding IDFs

        it draws the embedding vectors and corresponding IDFs from the
        (list of) term(s)

        Args:
            terms: (list of) term(s) whose embedding is to be drawn

        Returns:
            embedding vectors and IDFs corresponding to the terms
        """
        if isinstance(terms, str):
            return self._get_embs_idf([terms])

        # get terms and tokens that are "healthy"
        healthy, token_ids, terms_ary = self.__get_healthy_terms(terms)

        # get idfs
        idfs = self.__fetch_idfs(np.array(terms_ary))

        # get embs
        embs = self.__get_embs(terms_ary)

        return embs, idfs, healthy

    def predict_scores(
        self,
        new_doc_batch: list[str],
    ) -> list[dict[str, npt.NDArray[np.float64]]]:
        """ predict estimated simliarity between text and concepts

        similarity esimtation between text and each concept is computed.

        Args:
            new_doc_batch: given new batch of texts.

        Returns:
            esimated simliarity scores per concpet, per text.
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
