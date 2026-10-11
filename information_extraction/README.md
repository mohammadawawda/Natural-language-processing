# Information extraction

Finds (subject, relation, object) triplets in the Wikipedia pages of Donald Trump, Ruth Bader Ginsburg and J. K. Rowling in three ways:

* from the POS tags: two proper nouns in a row with a verb between them
* from the dependency tree: the subject and the direct object of the same verb, or the subject of a verb and the object of its preposition
* with Gemini, using the prompt that is in `information_extraction.py`

## How to run

```
python -m spacy download en_core_web_sm
export GEMINI_API_KEY=your_key
python information_extraction.py
```
