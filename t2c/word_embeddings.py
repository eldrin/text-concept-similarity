from typing import Optional, Union
import os
import logging
from pathlib import Path
import pickle

import numpy as np
import numpy.typing as npt

import fire
import h5py
from tokenizers import Tokenizer

import gensim
from gensim.models import KeyedVectors
import gensim.downloader as api
from gloves.model import GloVe


# it fetches the name of word-embeddings available from
# `gensim-data`.
available_wordembs = set([
    m for m in api.info(name_only=True)['models']
    if not m.startswith('__')
])
logger = logging.getLogger('text2conceptsim')


class Word2VecLookup:
    """

    It is a custom Word2VecLookup object based on h5py for a efficient prediction.
    The implementation is directly adopted from `here`_.

    The main benefit of using HDF as the main storage for the word-embedding is
    its fast access to the embeddings in out-of-core. Loading the large embeddings
    onto the memory takes substantially long time, which is not suitable for
    the resources are allocated dynamically per batch of calls.
    Indexing and fetching values from HDF (with a few restriction) can makes
    each prediction slower, while much more benefitial when the number of prediction
    within batch is not that large, while the total embedding size is big.

    Attributes:
        h5file (str): filename to the HDF file where the vector values are stored.
        lookupfile (str): filename to the pickle file which contains token-index
                          dictionary.
        lookup (dict[str, int]): the actual dictionary contains the token-index map.

    .. _here: https://gist.github.com/mynameisfiber/960ccae07daa2d891df9f88bfd7e3fbe
    """
    def __init__(self, dbpath):
        self.h5file = os.path.join(dbpath, "db.h5py")
        self.lookupfile = os.path.join(dbpath, "lookup.pkl")
        if not (os.path.exists(self.h5file) and
                os.path.exists(self.lookupfile)):
            logger.warning(
                "Word2VecLookup directory is malformed. Please recreate "
                "using Word2VecLookup.create_db"
            )
            raise TypeError
        with open(self.lookupfile, 'rb') as fd:
            self.lookup = pickle.load(fd)

    def __getitem__(
        self,
        items: Union[Union[str, bytes], list[Union[str, bytes]]]
    ) -> npt.NDArray[Union[np.float64, np.float32]]:
        """ get embedding vectors

        Args:
            items: input words to be indexed

        Return:
            word embedding vectors corresponding to the items
        """
        if isinstance(items, (str, bytes)):
            return self.__getitem__([items])
        w2v_indices = list(
            filter(
                lambda x: x is not None,
                map(
                    self.lookup.get,
                    items
                )
            )
        )

        w2v_indices_sort = sorted(set(w2v_indices))
        with h5py.File(self.h5file, 'r') as f:
            vectors = f['word2vec'][w2v_indices_sort]
        unsort = {w: i for i, w in enumerate(w2v_indices_sort)}
        unsort_idxs = [unsort[i]for i in w2v_indices if i in unsort]
        return np.array(vectors[unsort_idxs, ...])

    def __contains__(
        self,
        word: Union[str, bytes]
    ) -> bool:
        """ check whether the word is in the embedding or not.

        Args:
            word: a word to be checked

        Returns:
            boolean indicating whether the word is in the embedding or not.
        """
        return word in self.lookup

    def get_vector(
        self,
        word: str
    ) -> npt.ArrayLike:
        """ simple wrapper for __getitem__ for a single word

        this is mainly for the API match with other word-embedding models

        Args:
            word: input words to be indexed

        Returns:
            an embedding corresponding to the word
        """
        v = self.__getitem__([word])[0]
        return v

    def get_index(
        self,
        word: Union[str, bytes]
    ) -> int:
        """ get index of the term

        Args:
            word: the word to be indexed

        Returns:
            the integer index of the word
        """
        return self.lookup.get(word)

    @property
    def dtype(self) -> npt.DTypeLike:
        """ get the (numpy) data type of the embedding

        Returns:
            dtype of the embedding
        """
        with h5py.File(self.h5file, 'r') as f:
            dtype = f['word2vec'].dtype
        return dtype

    @property
    def vector_size(self) -> int:
        """ get the dimensionality of the embedding vectors

        Returns:
            the dimensionality of the embedding vectors
        """
        size = -1
        with h5py.File(self.h5file, 'r') as f:
            size = f['word2vec'].shape[1]
        return size

    @staticmethod
    def _check_valid(
        dbpath: str
    ) -> bool:
        """ check whether the dumped embedding files are valid

        it checks whether the saved embedding files are suitable for this
        class or not.

        Args:
            dbpath: path where the embedding files (db.h5 and lookup.pkl)

        Returns:
            boolean indicating whether it's valid or not.
        """
        res = False
        dbpath_ = Path(dbpath)
        if dbpath_.exists():
            h5_path = dbpath_ / 'db.h5py'
            lookup_path = dbpath_ / 'lookup.pkl'
            if h5_path.exists() and lookup_path.exists():
                res = True
        return res

    @staticmethod
    def create_db(
        word2vec_path: str,
        dbpath: str,
        binary: bool = False,
        is_glove: bool = True
    ) -> None:
        """ convert saved :obj:`gensim` embedding files into custom hdf files

        it converts the :obj:`gensim.models.KeyedVectors` compatible embedding
        dumps into the HDF based data file (and the lookup dictionary)

        Args:
            word2vec_path: path to the :obj:`gensim.models.KeyedVectors`
                           compatible dump of word-embeddings
            dbpath: path where the output is stored
            binary: indicates whether the source file is stored in binary
            is_glove: indicates whether the source file is saved in `GloVe`_
                      format.

        .. _GloVe: https://nlp.stanford.edu/projects/glove/
        """
        if gensim is None:
            logger.warning(
                "Cannot create h5db from word2vec binary file "
                "without gensim installed"
            )
        model = KeyedVectors.load_word2vec_format(word2vec_path,
                                                  binary=binary,
                                                  no_header=is_glove)

        os.makedirs(dbpath, exist_ok=True)
        h5file = os.path.join(dbpath, "db.h5py")
        lookupfile = os.path.join(dbpath, "lookup.pkl")

        lookup = dict(model.key_to_index.items())
        with open(lookupfile, 'wb+') as fd:
            pickle.dump(lookup, fd)

        with h5py.File(h5file, 'w') as f:
            f.create_dataset("word2vec", data=model.vectors)


class WordEmbedding:
    """
    It is a base class for the word-embedding interface which implements
    and defines some of the methods and members that are used in its child
    classes.

    Attributes:
        w2v_name_or_path (str): path to the word embedding dump file or the
                                unique name that can be used to load the
                                embedding from some platform (i.e.,
                                :obj:`gensim`.)
        w2v (object): core word embedding object. Specific type depends on which
                      embedding class is used as the core.
        _tokenizer (:obj:`tokenizers.Tokenizer`):
            the tokenizer used for preprocessing texts.
    """
    def __init__(
        self,
        w2v_name_or_path: str,
        tokenizer: Optional[Tokenizer] = None,
        is_glove: bool = False,
        is_gensim_model: bool = False,
        binary: bool = False
    ):
        self.w2v_name_or_path = w2v_name_or_path
        self._tokenizer = tokenizer
        self._load_w2v(
            w2v_name_or_path=w2v_name_or_path,
            tokenizer=tokenizer,
            is_glove=is_glove,
            is_gensim_model=is_gensim_model,
            binary=binary
        )

    def _load_w2v(
        self,
        w2v_name_or_path: str,
        tokenizer: Tokenizer,
        is_glove: bool = False,
        is_gensim_model: bool = False,
        binary: bool = False
    ) -> None:
        """ loads the word embedding to the class

        Args:
            w2v_name_or_path: filename of the word embedding dump.
            tokenizer: tokenizer to be used for preprocessing.
            is_glove: indicates whether the source file is saved in `GloVe`_
                      format.
            is_gensim_model: True if the model is compatible with :obj:`gensim`.
                             False if it is either for :obj:`gloves` or
                             :obj:`~t2c.word_embeddings.Word2VecLookup`.
            binary: indicates whether the source file is stored in binary
        """
        raise NotImplementedError()

    @property
    def n_components(self) -> int:
        """ returns the dimensionality of the word embedding vectors

        Returns:
            the dimensionality of the word embedding vectors
        """
        raise NotImplementedError()

    @property
    def dtype(self) -> npt.DTypeLike:
        """ returns the (numpy) data type of the word embedding vectors

        Returns:
            the (numpy) data type of the word embedding vectors
        """
        raise NotImplementedError()

    def __contains__(
        self,
        word: str
    ) -> bool:
        """ check whether the word is in the embedding or not.

        Args:
            word: a word to be checked

        Returns:
            boolean indicating whether the word is in the embedding or not.
        """
        raise NotImplementedError()

    def __getitem__(
        self,
        words: Union[Union[str, bytes], list[Union[str, bytes]]]
    ) -> npt.ArrayLike:
        """ get embedding vectors

        Args:
            words: input words to be indexed

        Return:
            word embedding vectors corresponding to the items
        """
        if isinstance(words, (str, bytes)):
            return self.__getitem__([words])

        # initiate the output container
        output = np.zeros((len(words), self.n_components), dtype=self.dtype)

        # check existing tokens
        words_proc = [self._preproc(w) for w in words]
        idx_exists, words_exists = tuple(
            zip(*[(j, w) for j, w in enumerate(words_proc) if w in self])
        )

        # get vectors and asign them into output container
        output[np.array(idx_exists)] = self.get_vectors(words_exists)

        return output

    def _preproc(self, word: str) -> str:
        """ preprocess the words

        it processes the word inputs if needed for specific embedding types.

        Args:
            word: input word

        Returns:
            processed word
        """
        raise NotImplementedError()

    def get_id(self, word: str) -> Optional[int]:
        """ get index of given token, based on the tokenizer embedded

        Args:
            word: input token

        Returns:
            index of the given token. it returns None if it is not found.
        """
        raise NotImplementedError()

    def get_vector(self, word: str) -> Optional[npt.ArrayLike]:
        """ get an embedding vector for given word

        it internally calls __getitem__.

        Args:
            word: input single word

        Return:
            word embedding vector corresponding to the word
        """
        raise NotImplementedError()

    def get_vectors(self, words: list[str]) -> npt.ArrayLike:
        """ get an embedding vectors for given words

        it internally calls __getitem__.

        Args:
            word: input words

        Return:
            word embedding vectors corresponding to the words
        """
        raise NotImplementedError()


class Word2VecLookupEmbedding(WordEmbedding):
    """
    it wraps the :obj:`~t2c.word_embeddings.Word2VecLookup` as the
    core embedding object.

    Attributes:
        w2v_name_or_path (str): path to the word embedding dump file or the
                                unique name that can be used to load the
                                embedding from some platform (i.e.,
                                :obj:`gensim`.)
        w2v (object): core word embedding object. In this particular class it
                      uses :obj:`~t2c.word_embeddings.Word2VecLookup` as core.
        _tokenizer (:obj:`tokenizers.Tokenizer`):
            the tokenizer used for preprocessing texts.
    """
    def __init__(
        self,
        w2v_name_or_path: str,
        tokenizer: Optional[Tokenizer] = None,
    ):
        if tokenizer is None:
            raise ValueError('[ERROR] GensimWordEmbedding requires '
                             'huggingface tokenizer to initialize!')

        super().__init__(w2v_name_or_path, tokenizer, False, True)

    def _load_w2v(
        self,
        w2v_name_or_path: str,
        *args,
        **kwargs
    ):
        """ loads the word embedding to the class

        Args:
            w2v_name_or_path: filename of the word embedding dump.
            tokenizer: tokenizer to be used for preprocessing.
            is_glove: indicates whether the source file is saved in `GloVe`_
                      format.
            is_gensim_model: True if the model is compatible with :obj:`gensim`.
                             False if it is either for :obj:`gloves` or
                             :obj:`~t2c.word_embeddings.Word2VecLookup`.
            binary: indicates whether the source file is stored in binary
        """
        self.w2v = Word2VecLookup(w2v_name_or_path)

    @property
    def n_components(self) -> int:
        """ returns the dimensionality of the word embedding vectors

        Returns:
            the dimensionality of the word embedding vectors
        """
        return self.w2v.vector_size

    @property
    def dtype(self) -> npt.DTypeLike:
        """ returns the (numpy) data type of the word embedding vectors

        Returns:
            the (numpy) data type of the word embedding vectors
        """
        return self.w2v.dtype

    def __contains__(
        self,
        word: str
    ) -> bool:
        """ check whether the word is in the embedding or not.

        Args:
            word: a word to be checked

        Returns:
            boolean indicating whether the word is in the embedding or not.
        """
        return word in self.w2v

    def _preproc(self, word: str) -> str:
        """ preprocess the words

        it processes the word inputs if needed for specific embedding types.

        Args:
            word: input word

        Returns:
            processed word
        """
        return word.replace('Ġ', '').replace(' ', '')

    def get_id(self, word: str) -> Optional[int]:
        """ get index of given token, based on the tokenizer embedded

        Args:
            word: input token

        Returns:
            index of the given token. it returns None if it is not found.
        """
        return self.w2v.get_index(self._preproc(word))

    def get_vector(self, word: str) -> Optional[npt.ArrayLike]:
        """ get an embedding vector for given word

        it internally calls __getitem__.

        Args:
            word: input single word

        Return:
            word embedding vector corresponding to the word
        """
        word = self._preproc(word)
        if word in self.w2v:
            return self.w2v.get_vector(word)
        else:
            return None

    def get_vectors(self, words: list[str]) -> npt.ArrayLike:
        """ get an embedding vectors for given words

        it internally calls __getitem__.

        Args:
            word: input words

        Return:
            word embedding vectors corresponding to the words
        """
        y = self.w2v[[self._preproc(w) for w in words]]
        return y


class GensimWordEmbedding(WordEmbedding):
    """
    it wraps the :obj:`gensim.models.KeyedVectors` as the
    core embedding object. It loads the embedding on the memory, which can
    take quite a bit of time if the embedding size is big.

    Attributes:
        w2v_name_or_path (str): path to the word embedding dump file or the
                                unique name that can be used to load the
                                embedding from some platform (i.e.,
                                :obj:`gensim`.)
        w2v (object): core word embedding object. In this particular class it
                      uses :obj:`gensim.models.KeyedVectors` as core.
        _tokenizer (:obj:`tokenizers.Tokenizer`):
            the tokenizer used for preprocessing texts.
    """
    def __init__(
        self,
        w2v_name_or_path: str,
        tokenizer: Optional[Tokenizer] = None,
        binary: bool = False,
        is_glove: bool = False
    ):
        if tokenizer is None:
            raise ValueError('[ERROR] GensimWordEmbedding requires '
                             'huggingface tokenizer to initialize!')

        super().__init__(w2v_name_or_path,
                         tokenizer,
                         is_glove,
                         binary)

    def _load_w2v(
        self,
        w2v_name_or_path: str,
        is_glove: bool = False,
        binary: bool = False,
        *args,
        **kwargs
    ):
        """
        """
        if w2v_name_or_path in available_wordembs:
            return api.load(w2v_name_or_path)

        self.w2v = KeyedVectors.load_word2vec_format(w2v_name_or_path,
                                                     binary=binary,
                                                     no_header=is_glove)

    def __contains__(
        self,
        word: str
    ) -> bool:
        """ check whether the word is in the embedding or not.

        Args:
            word: a word to be checked

        Returns:
            boolean indicating whether the word is in the embedding or not.
        """
        return word in self.w2v

    @property
    def n_components(self) -> int:
        """ returns the dimensionality of the word embedding vectors

        Returns:
            the dimensionality of the word embedding vectors
        """
        return self.w2v.vector_size

    @property
    def dtype(self) -> npt.DTypeLike:
        """ returns the (numpy) data type of the word embedding vectors

        Returns:
            the (numpy) data type of the word embedding vectors
        """
        return self.w2v.vectors.dtype

    def _preproc(self, word: str) -> str:
        """ preprocess the words

        it processes the word inputs if needed for specific embedding types.

        Args:
            word: input word

        Returns:
            processed word
        """
        return word.replace('Ġ', '').replace(' ', '')

    def get_id(self, word: str) -> Optional[int]:
        """ get index of given token, based on the tokenizer embedded

        Args:
            word: input token

        Returns:
            index of the given token. it returns None if it is not found.
        """
        word = self._preproc(word)
        if word in self.w2v:
            return self.w2v.get_index(word)
        else:
            return None

    def get_vector(self, word: str) -> Optional[npt.ArrayLike]:
        """ get an embedding vector for given word

        it internally calls __getitem__.

        Args:
            word: input single word

        Return:
            word embedding vector corresponding to the word
        """
        word = self._preproc(word)
        if word in self.w2v:
            return self.w2v.get_vector(word)
        else:
            return None

    def get_vectors(self, words: list[str]) -> npt.ArrayLike:
        """ get an embedding vectors for given words

        it internally calls __getitem__.

        Args:
            word: input words

        Return:
            word embedding vectors corresponding to the words
        """
        words_proc = [self._preproc(w) for w in words]
        words_exists = [w for w in words_proc if w in self.w2v]
        return self.w2v[words_exists]


class GloVeWordEmbedding(WordEmbedding):
    """
    it wraps the :obj:`gloves.model.GloVe` as the
    core embedding object. It loads the embedding on the memory, which can
    take quite a bit of time if the embedding size is big.

    Attributes:
        w2v_name_or_path (str): path to the word embedding dump file or the
                                unique name that can be used to load the
                                embedding from some platform (i.e.,
                                :obj:`gensim`.)
        w2v (object): core word embedding object. In this particular class it
                      uses :obj:`gloves.model.GloVe` as core.
        _tokenizer (:obj:`tokenizers.Tokenizer`):
            the tokenizer used for preprocessing texts.
    """
    def __init__(
        self,
        w2v_name_or_path: str,
    ):
        """
        """
        super().__init__(w2v_name_or_path,
                         tokenizer = None,
                         is_glove = False,
                         binary = False)

    def _load_w2v(
        self,
        w2v_name_or_path: str,
        *args,
        **kwargs
    ):
        """
        """
        self.w2v = GloVe.from_file(w2v_name_or_path)
        # reset tokenizer with the one embedded in `self.w2v`
        self._tokenizer = self.w2v._tokenizer

    @property
    def n_components(self) -> int:
        """ returns the dimensionality of the word embedding vectors

        Returns:
            the dimensionality of the word embedding vectors
        """
        return self.w2v.n_components

    @property
    def dtype(self) -> npt.DTypeLike:
        """ returns the (numpy) data type of the word embedding vectors

        Returns:
            the (numpy) data type of the word embedding vectors
        """
        return self.w2v.dtype

    def __contains__(self, word: str) -> bool:
        """ check whether the word is in the embedding or not.

        Args:
            word: a word to be checked

        Returns:
            boolean indicating whether the word is in the embedding or not.
        """
        return self.w2v.get_id(word) is not None

    def _preproc(self, word: str) -> str:
        """ preprocess the words

        it is an identity function.
        (word embedding will be directly recognize the raw tokenized input)

        Args:
            word: input word

        Returns:
            processed word
        """
        return word

    def get_id(self, word: str) -> Optional[int]:
        """ get index of given token, based on the tokenizer embedded

        Args:
            word: input token

        Returns:
            index of the given token. it returns None if it is not found.
        """
        return self.w2v.get_id(word)

    def get_vector(self, word: str) -> Optional[npt.ArrayLike]:
        """ get an embedding vector for given word

        it internally calls __getitem__.

        Args:
            word: input single word

        Return:
            word embedding vector corresponding to the word
        """
        return self.w2v.get_vector(word)

    def get_vectors(self, words: list[str]) -> npt.ArrayLike:
        """ get an embedding vectors for given words

        it internally calls __getitem__.

        Args:
            word: input words

        Return:
            word embedding vectors corresponding to the words
        """
        words_exists = [w for w in words if self.w2v.get_id(w) is not None]
        return np.array([self.w2v.get_vector(w) for w in words_exists])


def load_word_embs(
    path_or_name: str,
    tokenizer: Optional[Tokenizer]=None,
    gensim_model: bool=True,
    is_glove: bool=False,
    binary: bool=False
) -> WordEmbedding:
    """ loads the word embedding file to the word embedding objects

    Args:
        path_or_name: path to the word embedding dump file or the
                      unique name that can be used to load the
                      embedding from some platform (i.e.,
                      :obj:`gensim`.)
        tokenizer: the tokenizer used for preprocessing texts.
        gensim_model: True if the model is compatible with :obj:`gensim`.
                      False if it is either for :obj:`gloves` or
                      :obj:`~t2c.word_embeddings.Word2VecLookup`.
        is_glove: indicates whether the source file is saved in `GloVe`_
                  format.
        binary: indicates whether the source file is stored in binary

    Returns:
        a instance of :obj:`t2c.word_embeddings.WordEmbedding` object.

    .. _GloVe: https://nlp.stanford.edu/projects/glove/
    """
    if gensim_model:
        if tokenizer is None:
            raise ValueError('[ERROR] GensimWordEmbedding requires '
                             'huggingface tokenizer to initialize!')
        if Word2VecLookup._check_valid(path_or_name):
            return Word2VecLookupEmbedding(path_or_name, tokenizer)
        return GensimWordEmbedding(path_or_name, tokenizer, binary, is_glove)
    else:
        return GloVeWordEmbedding(path_or_name)


def gensim2hdf(
    w2v_path: str,
    db_path: str,
    binary: bool,
    is_glove: bool
) -> None:
    """ converts word-embedding into h5py file formats

    it converts word-embedding files that are compatible with `gensim` models
    into the h5py file format (and separate lookup table for tokens in pickle binary)

    Args:
        w2v_path: path to the word embedding file to be converted
        db_path: path the converted files saved
        binary: indicates whether the origin file is binary file (True) or text (False)
        is_glove: indicates whether the origin file is glove file (True / no header)
                  or typical gensim keyedvector file (False / header)

    """
    Word2VecLookup.create_db(w2v_path, db_path, binary, is_glove)


def main():
    fire.Fire(gensim2hdf)
