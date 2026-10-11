# Language models

`language_models.py` trains a unigram and a bigram language model on the lemmas of Wikitext 2 (every line is a document and starts with START), and then:

* finds the most likely word after "I have a house in"
* computes the log probability of two sentences and their perplexity with the bigram model
* does the same with a linear interpolation of the two models (2/3 bigram and 1/3 unigram)

## How to run

```
python -m spacy download en_core_web_sm
python language_models.py
```
