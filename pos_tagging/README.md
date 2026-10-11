# POS tagging

`pos_tagging.py` tags the news part of the Brown corpus, the last 10% of the sentences is the test set:

* b: the most likely tag of every word, unknown words get NN
* c: a bigram HMM with maximum likelihood and the Viterbi algorithm
* d: the same with add one smoothing for the emissions
* e: pseudo words for rare and unknown words, with maximum likelihood and with add one smoothing, and the most common mistakes from the confusion matrix

## Results

Error rates on the test set:

* most likely tag: total 0.148 (known words 0.070, unknown words 0.750)
* HMM: total 0.281 (known 0.215, unknown 0.791)
* HMM with add one: total 0.210 (known 0.146, unknown 0.713)
* HMM with pseudo words: total 0.243 (known 0.201, unknown 0.570)
* HMM with pseudo words and add one: total 0.194 (known 0.148, unknown 0.552)

## How to run

```
python pos_tagging.py
```
