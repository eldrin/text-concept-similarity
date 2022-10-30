Text to Concept Similarity Estimates
====================================

It is a small pakcage that implements a few procedures that estimates similarities between the given text and a certain arbitrary concept. For instance, a text of auto-essay and the similarities towards high level concepts such as traits of personality. (i.e., Honesty-Humility, Emotionality, Extraversion, Agreeableness, ...) In natural language processing (NLP), it is common practice that estimating the "distance" or "similarity" between two sets of text inputs using word-embeddings (or some more advanced means such as [Transformers](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))). Thus, this package aims to implement some of the reliable ways to estimate the similarity between a text passage and a group of words using some of those techniques.

Ultimately, we seek to build a tool that can be used in general scenario where one would like to estimate the similarities from given texts to the arbitrary sets of concepts.

Currently, two estimating procedures are implemented:

- a frequency based method relying on the exact string-match ([Ponizovskiy et al. 2020](https://onlinelibrary.wiley.com/doi/full/10.1002/per.2294)).
- estimating similarities based on the cosine distance among word embeddings weighted by the inverse document frequencies.


## Installation

Currently we provide the package through only by the github.

```bash
pip install git+https://github.com/eldrin/text-concept-similarity@main
```


## Usage

Once installed, the package can be used as the normal python package. The most important objects would be the [`estimators`](https://jaehun.kim/text-concept-similarity/t2c.html#module-t2c.estimator).

```python
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
```

Also, we provide the CLI outlet for extracting the estimations directly from the commandline.

```bash
t2csim \
    /path/to/input_text.txt \
    /path/to/output.csv \
    /path/to/embedding/or/embedding_name/ \
    --dict_fn=/path/to/dictionary.json
```

It expect the input text file contianing text files where each "document" corresponds to each row.

```text
this is the first document.
this is the second document.
...
```

Without specifying the dictionary file for concept words, the personal value dictionary ([Ponizovskiy et al. 2020](https://onlinelibrary.wiley.com/doi/full/10.1002/per.2294)) is used as the default dictionary. The custom dictionary can be provided by a text file containing the dictionary in `json` format.


```json
{
    "concept1": ["concept1_word1", "concept1_word2", ...],
    "concept2": ["concept2_word1", "concept2_word2", ...],
    ...
}
```

For more detailed information, check `t2csim --help`.

With [`docker`](https://www.docker.com/) and [`cog`](https://github.com/replicate/cog), one can also call the extraction within the docker container easily.

It requires two steps. Firstly, the embedding files should be fetched.

```bash
TBD
```

and then the extraction can be called by the command

```bash
cog predict \
    -i text=@path/to/input_text.txt \
    -i word_embs="word2vec-google-news-300" \
    -i dictionary=@path/to/custom_dictionary.json \
    -i normalization="softmax" \
    -i alpha=0.5
```

## Authors

Jaehun Kim (jaehun_dot_j_dot_kim_at_gmail_dot_com)


## License

MIT


## Citation

```
{bibtex_tbd}
```

## TODOs

- [ ] test coverage
  - [ ] data I/O
  - [ ] utility functions
  - [ ] estimators
    - [x] wordcounting
    - [ ] further corner cases on embedding similarity
    - [ ] further corner cases on word counting
- [x] docstrings
- [ ] improve docstrings
- [ ] further optimization


## Acknowledge
