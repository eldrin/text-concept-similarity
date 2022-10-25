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
# `gensi-data`.
available_wordembs = set([
    m for m in api.info(name_only=True)['models']
    if not m.startswith('__')
])
logger = logging.getLogger('text2conceptsim')


class Word2VecLookup:
    """ custom Word2VecLookup object based on h5py

    adopted from https://gist.github.com/mynameisfiber/960ccae07daa2d891df9f88bfd7e3fbe

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

    def __getitem__(self, items) -> npt.ArrayLike:
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
        return vectors[unsort_idxs, ...]

    def __contains__(self, word: str) -> bool:
        """
        """
        return word in self.lookup

    def get_vector(
        self,
        word: str
    ) -> npt.ArrayLike:
        """ simple wrapper for __getitem__ for a single word

        this is mainly for the API match with other word-embedding models
        """
        v = self.__getitem__([word])[0]
        return v

    def get_index(
        self,
        word: str
    ) -> int:
        """
        """
        return self.lookup.get(word)

    @property
    def dtype(self):
        """
        """
        with h5py.File(self.h5file, 'r') as f:
            dtype = f['word2vec'].dtype
        return dtype

    @property
    def vector_size(self) -> int:
        """
        """
        size = -1
        with h5py.File(self.h5file, 'r') as f:
            size = f['word2vec'].shape[1]
        return size

    @staticmethod
    def _check_valid(
        dbpath: str
    ) -> bool:
        """
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
    ):
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
    """
    def __init__(
        self,
        w2v_name_or_path: str,
        tokenizer: Optional[Tokenizer] = None,
        is_glove: bool = False,
        is_gensim_model: bool = False,
        binary: bool = False
    ):
        """
        """
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
    ):
        """
        """
        raise NotImplementedError()

    @property
    def n_components(self):
        """
        """
        raise NotImplementedError()

    @property
    def dtype(self):
        """
        """
        raise NotImplementedError()

    def __contains__(
        self,
        word: str
    ) -> bool:
        """
        """
        raise NotImplementedError()

    def __getitem__(
        self,
        words: Union[Union[str, bytes], list[Union[str, bytes]]]
    ) -> npt.ArrayLike:
        """
        """
        if isinstance(words, (str, bytes)):
            return self.__getitem__([words])

        # initiate the output container
        output = np.zeros((len(words), self.n_components), dtype=self.dtype)
        # output = np.random.randn(len(words), self.n_components)

        # check existing tokens
        words_proc = [self._preproc(w) for w in words]
        idx_exists, words_exists = tuple(
            zip(*[(j, w) for j, w in enumerate(words_proc) if w in self])
        )

        # get vectors and asign them into output container
        output[np.array(idx_exists)] = self.get_vectors(words_exists)

        return output

    def _preproc(self, word: str) -> str:
        """
        """
        raise NotImplementedError()

    def get_id(self, word: str) -> Optional[int]:
        """
        """
        raise NotImplementedError()

    def get_vector(self, word: str) -> Optional[npt.ArrayLike]:
        """
        """
        raise NotImplementedError()

    def get_vectors(self, words: list[str]) -> npt.ArrayLike:
        """
        """
        raise NotImplementedError()

    def encode(self, word_or_sentence: str) -> tuple[np.ndarray, list[int]]:
        """
        """
        raise NotImplementedError()


class Word2VecLookupEmbedding(WordEmbedding):
    """
    """
    def __init__(
        self,
        w2v_name_or_path: str,
        tokenizer: Optional[Tokenizer] = None,
    ):
        """
        """
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
        """
        """
        # if tokenizer is None:
        #     raise ValueError('[ERROR] GensimWordEmbedding requires '
        #                      'huggingface tokenizer to initialize!')
        # it could be either `Word2VecLookup` or `gensim.models.KeyedVectors`
        self.w2v = Word2VecLookup(w2v_name_or_path)

    @property
    def n_components(self):
        """
        """
        return self.w2v.vector_size

    @property
    def dtype(self):
        """
        """
        return self.w2v.dtype

    def __contains__(
        self,
        word: str
    ) -> bool:
        """
        """
        return word in self.w2v

    def _preproc(self, word: str) -> str:
        """
        """
        return word.replace('Ġ', '').replace(' ', '')

    def get_id(self, word: str) -> Optional[int]:
        """
        """
        return self.w2v.get_index(self._preproc(word))

    def get_vector(self, word: str) -> Optional[npt.ArrayLike]:
        """
        """
        word = self._preproc(word)
        if word in self.w2v:
            return self.w2v.get_vector(word)
        else:
            return None

    def get_vectors(self, words: list[str]) -> npt.ArrayLike:
        y = self.w2v[[self._preproc(w) for w in words]]
        return y


class GensimWordEmbedding(WordEmbedding):
    """
    """
    def __init__(
        self,
        w2v_name_or_path: str,
        tokenizer: Optional[Tokenizer] = None,
        binary: bool = False,
        is_glove: bool = False
    ):
        """
        """
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
        """
        """
        return word in self.w2v

    @property
    def n_components(self):
        """
        """
        return self.w2v.vector_size

    @property
    def dtype(self):
        """
        """
        return self.w2v.vectors.dtype

    def _preproc(self, word: str) -> str:
        """
        """
        return word.replace('Ġ', '').replace(' ', '')

    def get_id(self, word: str) -> Optional[int]:
        """
        """
        word = self._preproc(word)
        if word in self.w2v:
            return self.w2v.get_index(word)
        else:
            return None

    def get_vector(self, word: str) -> Optional[npt.ArrayLike]:
        """
        """
        word = self._preproc(word)
        if word in self.w2v:
            return self.w2v.get_vector(word)
        else:
            return None

    def get_vectors(self, words: list[str]) -> npt.ArrayLike:
        words_proc = [self._preproc(w) for w in words]
        words_exists = [w for w in words_proc if w in self.w2v]
        return self.w2v[words_exists]

    # def encode(self, word_or_sentence: str) -> tuple[np.ndarray, list[int]]:
    #     """
    #     """
    #     tok = self._tokenizer.encode(word_or_sentence)
    #     vecs = []
    #     ids = []
    #     for i, token in zip(tok.ids, tok.tokens):
    #         token = token.replace('Ġ', '')
    #         if token in self.w2v:
    #             w = self.w2v[token]
    #             vecs.append(w)
    #             ids.append(i)
    #     if len(vecs) == 0:
    #         logger.warning('No vectors are found from this input! returning None...')

    #     # concatenate
    #     vecs = np.stack(vecs)

    #     return vecs, ids


class GloVeWordEmbedding(WordEmbedding):
    """
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
    def n_components(self):
        """
        """
        return self.w2v.n_components

    @property
    def dtype(self):
        """
        """
        return self.w2v.dtype

    def __contains__(self, word: str) -> bool:
        """
        """
        return self.w2v.get_id(word) is not None

    def _preproc(self, word: str) -> str:
        """

        identity (word embedding will be directly recognize the raw tokenized input)
        """
        return word

    def get_id(self, word: str) -> Optional[int]:
        """
        """
        return self.w2v.get_id(word)

    def get_vector(self, word: str) -> Optional[npt.ArrayLike]:
        """
        """
        return self.w2v.get_vector(word)

    def get_vectors(self, words: list[str]) -> npt.ArrayLike:
        words_exists = [w for w in words if self.w2v.get_id(w) is not None]
        return np.array([self.w2v.get_vector(w) for w in words_exists])


def load_word_embs(
    path_or_name: str,
    tokenizer: Optional[Tokenizer]=None,
    gensim_model: bool=True,
    is_glove: bool=False,
    binary: bool=False
) -> WordEmbedding:
    """
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
    """converts word-embedding into h5py file formats

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
