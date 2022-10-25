from os.path import join

import pkg_resources


DEFAULT_TOKENIZER_FN = 'tokenizer-wiki210720.json'
DEFAULT_IDF_FN = 'wikidump20210720_idf.txt'
TEST_INPUT_FN = 'test_input.txt'


def default_tokenizer():
    """ read the filename of pre-trained tokenizer
    """
    return pkg_resources.resource_filename(
        __name__, join('data', DEFAULT_TOKENIZER_FN)
    )


def default_idf():
    """ read the filename of pre-trained tokenizer
    """
    return pkg_resources.resource_filename(
        __name__, join('data', DEFAULT_IDF_FN)
    )
