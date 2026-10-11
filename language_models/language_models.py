import math

import spacy
from datasets import load_dataset
from collections import Counter, defaultdict


print("Loading Spacy and datasets...")
nlp = spacy.load("en_core_web_sm")
text = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
print("Datasets loaded.")


def get_lemmas(line):
    # lemmas of the words only, without punctuation and numbers
    return [token.lemma_ for token in nlp(line) if token.is_alpha]


def load_local_db_from_dataset():
    """
    every line of the dataset is a document and every document starts with START
    returns all the lemmas in one list
    """
    print("Loading db...")
    db = []
    for i in range(text.num_rows):
        if text[i]['text']:
            lemmas = get_lemmas(text[i]['text'])
            if lemmas:
                db.extend(["START"] + lemmas)
        print(f"Line {i} loaded.")
    print("\nLoading Complete!")
    return db


db = load_local_db_from_dataset()


def initialize_unigram_model(db, l=1):
    """
    unigram probabilities, START is not a word so it's not counted
    db: list of lemmas
    l: every probability is multiplied by it (used for the interpolation)
    """
    unigram_dict = defaultdict(float)
    words = [word for word in db if word != 'START']
    total_words = len(words)
    word_counts = Counter(words)

    for word, count in word_counts.items():
        unigram_dict[word] = l * (count / total_words)

    return unigram_dict


def initialize_bigram_model(db, l=1):
    """
    bigram probabilities p(word2 | word1), pairs that go into the next document are skipped
    db: list of lemmas
    l: every probability is multiplied by it (used for the interpolation)
    """
    bigram_dict = defaultdict(float)
    word_counts = Counter(db)
    bigram_counts = Counter(zip(db, db[1:]))

    for (word1, word2), count in bigram_counts.items():
        if word2 != 'START':
            bigram_dict[(word1, word2)] = l * (count / word_counts[word1])

    return bigram_dict


def find_most_likely_next_word(sentence, bigram_dict):
    """
    the word with the highest bigram probability after the last word of the sentence
    """
    last_word = get_lemmas(sentence)[-1]
    best_next_word, max_probability = None, 0

    for (word1, word2), probability in bigram_dict.items():
        if word1 == last_word and probability > max_probability:
            max_probability, best_next_word = probability, word2

    return best_next_word, max_probability


print("\nNext word prediction:")
bigram_dict = initialize_bigram_model(db)
sentence = "I have a house in"
next_word, probability = find_most_likely_next_word(sentence, bigram_dict)

if next_word:
    print(f"The most likely next word after '{sentence}' is '{next_word}' with a probability of {probability:.4f}")
else:
    print(f"No suitable next word found after '{sentence}'")


def calculate_sentence_probability(sentence, models):
    """
    log probability of a sentence, the first word comes after START
    models: [bigram] for the bigram model, or [bigram, unigram] for the interpolation
    in the interpolation both models are already multiplied by their lambda so we just add them
    """
    words = ["START"] + get_lemmas(sentence)
    log_probability = 0

    for i in range(len(words) - 1):
        probability = models[0].get((words[i], words[i + 1]), 0)
        if len(models) == 2:
            probability += models[1].get(words[i + 1], 0)
        if probability == 0:
            return -math.inf
        log_probability += math.log(probability)

    return log_probability


print("\nSentence probabilities:")
first_sentence = "Brad Pitt was born in Oklahoma"
second_sentence = "The actor was born in USA"
print(f"Log probability of '{first_sentence}': {calculate_sentence_probability(first_sentence, [bigram_dict])}")
print(f"Log probability of '{second_sentence}': {calculate_sentence_probability(second_sentence, [bigram_dict])}")


def calculate_perplexity(sentences, models):
    """
    perplexity of all the sentences together as one test set
    M is the number of words in all of them, START is not counted
    """
    log_probability_sum, word_count = 0, 0

    for sentence in sentences:
        log_probability_sum += calculate_sentence_probability(sentence, models)
        word_count += len(get_lemmas(sentence))

    return math.exp(-log_probability_sum / word_count)


print("\nPerplexity:")
sentences = [first_sentence, second_sentence]
print(f"Perplexity of the bigram model: {calculate_perplexity(sentences, [bigram_dict])}")

print("\nLinear interpolation:")
unigram_parameter = 1 / 3
bigram_parameter = 2 / 3
interpolated_models = [initialize_bigram_model(db, bigram_parameter), initialize_unigram_model(db, unigram_parameter)]

print(f"Interpolated log probability of '{first_sentence}': "
      f"{calculate_sentence_probability(first_sentence, interpolated_models)}")
print(f"Interpolated log probability of '{second_sentence}': "
      f"{calculate_sentence_probability(second_sentence, interpolated_models)}")
print(f"Perplexity of the interpolated model: {calculate_perplexity(sentences, interpolated_models)}")
