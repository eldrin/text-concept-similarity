from cog import BasePredictor, Input, Path
from t2c.__main__ import extract


WORD_EMBEDDING_PATHS = {
    'word2vec-google-news-300': './data/embeddings/word2vec-google-news-300/',
    'glove.840B.300d': './data/embeddings/glove.840B.300d/',
    'mxm-faruqui-0': './data/embeddings/faruqui_eval/0/output',
    'mxm-faruqui-1': './data/embeddings/faruqui_eval/1/output',
    'mxm-faruqui-2': './data/embeddings/faruqui_eval/2/output',
    'mxm-faruqui-3': './data/embeddings/faruqui_eval/3/output',
    'mxm-faruqui-4': './data/embeddings/faruqui_eval/4/output',
    'mxm-faruqui-5': './data/embeddings/faruqui_eval/5/output',
    'mxm-faruqui-6': './data/embeddings/faruqui_eval/6/output',
    'mxm-faruqui-7': './data/embeddings/faruqui_eval/7/output',
    'mxm-faruqui-8': './data/embeddings/faruqui_eval/8/output',
    'mxm-faruqui-9': './data/embeddings/faruqui_eval/9/output',
    'mxm-cv-0': './data/embeddings/split_eval/0/output',
    'mxm-cv-1': './data/embeddings/split_eval/1/output',
    'mxm-cv-2': './data/embeddings/split_eval/2/output',
    'mxm-cv-3': './data/embeddings/split_eval/3/output',
    'mxm-cv-4': './data/embeddings/split_eval/4/output',
    'mxm-cv-5': './data/embeddings/split_eval/5/output',
    'mxm-cv-6': './data/embeddings/split_eval/6/output',
    'mxm-cv-7': './data/embeddings/split_eval/7/output',
    'mxm-cv-8': './data/embeddings/split_eval/8/output',
    'mxm-cv-9': './data/embeddings/split_eval/9/output',
    'wordcount': 'wordcount',
}

WORD_EMBEDDING_FLAGS = {
    'word2vec-google-news-300': {
        'is_gensim_model': True,
        'is_glove': False,
        'binary': True
    },
    'glove.840B.300d': {
        'is_gensim_model': True,
        'is_glove': True,
        'binary': True
    },
    'mxm-faruqui': {
        'is_gensim_model': False,
        'is_glove': True,
        'binary': True
    },
    'mxm-cv': {
        'is_gensim_model': False,
        'is_glove': True,
        'binary': True
    }
}


class Predictor(BasePredictor):
    """
    """
    def predict(
        self,
        text: Path = Input(description="text file contains documents per each line"),
        word_embs: str = Input(description="name of word embedding to be used",
                               choices=list(WORD_EMBEDDING_PATHS.keys())),
        dictionary: Path = Input(description=("json file contains dictionary "
                                        "for the concepts to be estimated"),
                                 default=None),
        normalization: str = Input(description="normalization method",
                                   choices=['zscore', 'softmax', 'l2', 'null'],
                                   default=None)
    ) -> Path:
        """ Run a single prediction on the model """
        # gen output name from input name? or temp filename?
        out_fn = text.parent / (text.stem + '_est.csv')

        # using extract() to compute the estimation
        emb_type = [
            k for k in WORD_EMBEDDING_FLAGS.keys()
            if word_embs.startswith(k)
        ][0]
        norm = None if normalization == 'null' else normalization
        dic = None if dictionary is None else dictionary.as_posix()
        extract(
            text.as_posix(),
            out_fn.as_posix(),
            WORD_EMBEDDING_PATHS[word_embs],
            dict_fn = dic,
            normalization = norm,
            **WORD_EMBEDDING_FLAGS[emb_type]
        )

        # output the path for the output text file
        return out_fn
