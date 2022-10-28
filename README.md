# Text to Concept Similarity Estimates

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

Once installed, the main score estimation function can be called directly from command line.

```bash
t2c /path/to/input_text.txt /path/to/output.csv /path/to/embedding/or/embedding_name/
```

With `[docker](https://www.docker.com/)` and `[cog](https://github.com/replicate/cog)`, one can also call the extraction within the docker container easily.

It requires two steps. Firstly, the embedding files should be fetched.

```bash
TBD
```

and then the extraction can be called by the command

```bash
cog predict \
    -i text=@/path/to/input_text.txt \
    -i word_embs="word2vec-google-news-300" \
    -i normalization="softmax" 
```

## Authors

Jaehun Kim


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
    - [ ] wordcounting
    - [ ] further corner cases on embedding similarity
- [ ] docstrings
- [ ] further optimization


## Acknowledge
