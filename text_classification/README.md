# Text classification

Classifies posts of 4 topics from the 20 newsgroups dataset with:

* a single layer (log linear) model on TFIDF vectors with 2000 features
* the same with one hidden layer of 500
* a finetuned distilroberta transformer

The two MLPs are trained on 10%, 20%, 50% and all of the train data, and the transformer on 10% and 20%. For every run it plots the train loss and the validation accuracy of every epoch and prints the number of trainable parameters.

## How to run

```
python text_classification.py
```
