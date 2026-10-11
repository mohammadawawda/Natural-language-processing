# Sentiment analysis

Positive or negative sentiment on the Stanford Sentiment Treebank with 4 models:

* a log linear model on the average one hot vector of the sentence
* a log linear model on the average word2vec vector
* a bidirectional LSTM on the word2vec vectors, 52 words per sentence
* a finetuned distilroberta transformer (`transformer.py`)

For every model it plots the train and validation loss and accuracy, and prints the test accuracy and the accuracy on two special parts of the test set: sentences with negated polarity and sentences with rare words.

`data_loader.py`, the loader of the treebank, is not my code.

## How to run

Put the `stanfordSentimentTreebank` folder next to the code, then:

```
python sentiment_analysis.py
python transformer.py
```

The first run downloads word2vec (1.6 GB) and keeps the vectors of our words in `w2v_dict.pkl`.
