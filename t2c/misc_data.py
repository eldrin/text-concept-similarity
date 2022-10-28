from os.path import join

import pkg_resources


DEFAULT_TOKENIZER_FN = 'tokenizer-wiki210720.json'
DEFAULT_IDF_FN = 'wikidump20210720_idf.txt'
TEST_INPUT_FN = 'test_input.txt'


def default_tokenizer() -> str:
    """ read the filename of pre-trained tokenizer

    it returns filename to the configuration of pretrained tokenizer
    which is provided as a default tokenizer. The tokenizer is trained
    from dump of English Wikipedia, snapshot from July 2021.

    Returns:
        filename to the configuration of pretrained tokenizer
        which is provided as a default tokenizer.
    """
    return pkg_resources.resource_filename(
        __name__, join('data', DEFAULT_TOKENIZER_FN)
    )


def default_idf() -> str:
    """ read the filename of pre-computed IDFs

    it returns filename to the pre-computed IDF values for the words
    , which is provided as a default IDFs. It is based on the default
    tokenizer, and IDF also is computed from the same snapshot of the
    English Wikipedia dump.

    Returns:
        filename to the pre-computed IDF values.
    """
    return pkg_resources.resource_filename(
        __name__, join('data', DEFAULT_IDF_FN)
    )
