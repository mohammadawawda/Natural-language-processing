from collections import defaultdict
from collections import Counter
import nltk
import re
import numpy as np

nltk.download('brown')
from nltk.corpus import brown


def extract_prefix_from_tag(tag):
    # the tag is the part before the first + or -, tags like "--" stay as they are
    prefix = re.split(r"[+-]", tag)[0]
    return prefix if prefix else tag


tagged_news_sentences = brown.tagged_sents(categories='news')

# the test set is the last 10% of the sentences
split = int(len(tagged_news_sentences) * 0.9)
tagged_training_set_with_symbols = tagged_news_sentences[:split]
tagged_test_set_with_symbols = tagged_news_sentences[split:]

tagged_training_set = []
for sentence in tagged_training_set_with_symbols:
    modified_sentence = [(word, extract_prefix_from_tag(tag)) for word, tag in sentence]
    tagged_training_set.append(modified_sentence)

tagged_test_set = []
for sentence in tagged_test_set_with_symbols:
    modified_sentence = [(word, extract_prefix_from_tag(tag)) for word, tag in sentence]
    tagged_test_set.append(modified_sentence)

# b(i) the most likely tag of every word in the training set
word_tags_count = defaultdict(Counter)
for sentence in tagged_training_set:
    for word, tag in sentence:
        word_tags_count[word][tag] += 1

word_most_likely_tag = {}
for word, tags_counter in word_tags_count.items():
    word_most_likely_tag[word] = tags_counter.most_common(1)[0][0]

# b(ii) error rates, unknown words get NN
known_words_correct = 0
known_words_total = 0
unknown_words_correct = 0
unknown_words_total = 0

for sentence in tagged_test_set:
    for word, true_tag in sentence:
        if word in word_most_likely_tag:
            known_words_total += 1
            if word_most_likely_tag[word] == true_tag:
                known_words_correct += 1
        else:
            unknown_words_total += 1
            if 'NN' == true_tag:
                unknown_words_correct += 1

known_word_accuracy = known_words_correct / known_words_total if known_words_total > 0 else 0
unknown_word_accuracy = unknown_words_correct / unknown_words_total if unknown_words_total > 0 else 0
all_words_total = known_words_total + unknown_words_total
total_accuracy = (known_words_correct + unknown_words_correct) / all_words_total if all_words_total > 0 else 0

known_word_error_rate = 1 - known_word_accuracy
unknown_word_error_rate = 1 - unknown_word_accuracy
total_error_rate = 1 - total_accuracy

print()
print("Most likely tag")
print()
print(f"Total Error Rate: {total_error_rate}")
print(f"Known Word Error Rate: {known_word_error_rate}")
print(f"Unknown Word Error Rate: {unknown_word_error_rate}")

# c(i) transition probabilities, every sentence starts with START and ends with STOP
tag_transitions = defaultdict(Counter)
tag_counts = Counter()

for sentence in tagged_training_set:
    tags = ["START"] + [tag for (_, tag) in sentence] + ["STOP"]
    for i in range(len(tags) - 1):
        tag_counts[tags[i]] += 1
        tag_transitions[tags[i]][tags[i+1]] += 1

tag_transition_probs = {}
for cur_tag, counts_of_all_following_tags in tag_transitions.items():
    tag_transition_probs[cur_tag] = {}
    for following_tag, count_of_following_tag in counts_of_all_following_tags.items():
        tag_transition_probs[cur_tag][following_tag] = count_of_following_tag / tag_counts[cur_tag]

# emission probabilities p(word | tag)
tag_emission_probabilities = defaultdict(dict)
for word, tag_counts_of_words in word_tags_count.items():
    for tag, count in tag_counts_of_words.items():
        tag_emission_probabilities[word][tag] = count / tag_counts[tag]


def viterbi_algorithm(sentence, tag_transition_probs, tag_emission_probabilities, tag_counts, unseen=None):
    """
    bigram viterbi, returns the best tag for every word in the sentence
    unseen: the smoothed emission of a (word, tag) pair that wasn't in the training set,
    without it unknown words just get NN
    """
    def emission(word, tag):
        if unseen is not None:
            return tag_emission_probabilities.get(word, {}).get(tag, unseen[tag])
        return tag_emission_probabilities.get(word, {"NN": 1}).get(tag, 0)

    def normalize(column):
        # scaling every column keeps long sentences from going to 0, the best path stays the same
        total = sum(column.values())
        if total > 0:
            for tag in column:
                column[tag] /= total

    states = [tag for tag in tag_counts.keys() if tag != "START"]
    V = [{}]
    path = {}
    for tag in states:
        V[0][tag] = tag_transition_probs["START"].get(tag, 0) * emission(sentence[0], tag)
        path[tag] = [tag]
    normalize(V[0])

    for t in range(1, len(sentence)):
        V.append({})
        newpath = {}
        # only tags with a chance of being on the best path
        alive = [tag for tag in states if V[t-1][tag] > 0]

        for cur_tag in states:
            e = emission(sentence[t], cur_tag)
            if e == 0 or not alive:
                V[t][cur_tag] = 0
                newpath[cur_tag] = path[cur_tag] + [cur_tag]
                continue
            (prob, state) = max((V[t-1][prev_tag] * tag_transition_probs[prev_tag].get(cur_tag, 0) * e, prev_tag)
                                for prev_tag in alive)
            V[t][cur_tag] = prob
            newpath[cur_tag] = path[state] + [cur_tag]

        normalize(V[t])
        path = newpath

    # the last tag also has to move to STOP
    (prob, state) = max((V[len(sentence) - 1][tag] * tag_transition_probs[tag].get("STOP", 0), tag)
                        for tag in states)
    return path[state]


def test_viterbi_accuracy_with_known_unknown(tagged_sentences, viterbi_func, transition_probs, emission_probs, counts,
                                            word_most_likely_tag, unseen=None, original_sentences=None):
    """
    runs viterbi on every sentence and counts the mistakes for known and unknown words
    unseen: passed to viterbi for add one smoothing
    original_sentences: the sentences before the pseudo words, used to know which words are unknown
    """
    correct_predictions_known = 0
    total_predictions_known = 0
    correct_predictions_unknown = 0
    total_predictions_unknown = 0

    predicted_tags_list = []

    for i, sentence in enumerate(tagged_sentences):
        words = [word for word, tag in sentence]
        actual_tags = [tag for word, tag in sentence]
        predicted_tags = viterbi_func(words, transition_probs, emission_probs, counts, unseen)
        predicted_tags_list.append(predicted_tags)

        real_words = [word for word, tag in original_sentences[i]] if original_sentences else words
        for word, actual_tag, predicted_tag in zip(real_words, actual_tags, predicted_tags):
            if word in word_most_likely_tag:
                total_predictions_known += 1
                if actual_tag == predicted_tag:
                    correct_predictions_known += 1
            else:
                total_predictions_unknown += 1
                if actual_tag == predicted_tag:
                    correct_predictions_unknown += 1

    accuracy_known = correct_predictions_known / total_predictions_known if total_predictions_known > 0 else 0
    error_rate_known = 1 - accuracy_known
    accuracy_unknown = correct_predictions_unknown / total_predictions_unknown if total_predictions_unknown > 0 else 0
    error_rate_unknown = 1 - accuracy_unknown

    total_correct_predictions = correct_predictions_known + correct_predictions_unknown
    total_predictions = total_predictions_known + total_predictions_unknown
    accuracy_total = total_correct_predictions / total_predictions if total_predictions > 0 else 0
    error_rate_total = 1 - accuracy_total

    return accuracy_known, error_rate_known, accuracy_unknown, error_rate_unknown, accuracy_total, error_rate_total, predicted_tags_list


def print_results(title, results):
    print()
    print(title)
    print()
    print("Accuracy Known: ", results[0],
          "\nError Rate Known: ", results[1],
          "\nAccuracy Unknown: ", results[2],
          "\nError Rate Unknown: ", results[3],
          "\nAccuracy Total: ", results[4],
          "\nError Rate Total: ", results[5])


# c(iii)
results_c = test_viterbi_accuracy_with_known_unknown(tagged_test_set, viterbi_algorithm, tag_transition_probs,
                                                     tag_emission_probabilities, tag_counts, word_most_likely_tag)
print_results("Bigram HMM", results_c)


def compute_emission_probabilities_with_laplace_smoothing(tagged_sentences, tag_counts, word_tags_count):
    """
    add one smoothing for the emissions
    returns the emissions of the pairs we saw, and for every tag the emission of a pair we didn't see
    the + 1 in the vocabulary size is for the unknown words
    """
    vocabulary_size = len(set(word for sentence in tagged_sentences for word, _ in sentence)) + 1

    tag_emission_probabilities = defaultdict(dict)
    for word, tags in word_tags_count.items():
        for tag, count in tags.items():
            tag_emission_probabilities[word][tag] = (count + 1) / (tag_counts[tag] + vocabulary_size)

    unseen = {tag: 1 / (tag_counts[tag] + vocabulary_size) for tag in tag_counts}
    return tag_emission_probabilities, unseen


# d(i), d(ii)
emission_probabilities_with_laplace_smoothing, unseen_emission = \
    compute_emission_probabilities_with_laplace_smoothing(tagged_training_set, tag_counts, word_tags_count)

results_di = test_viterbi_accuracy_with_known_unknown(tagged_test_set, viterbi_algorithm, tag_transition_probs,
                                                      emission_probabilities_with_laplace_smoothing, tag_counts,
                                                      word_most_likely_tag, unseen_emission)
print_results("Bigram HMM with add one smoothing", results_di)


def create_psuedo_tag_for_word(word):
    if (re.match(r'^\d{2}$', word) and 70 < int(word)) or (re.match(r'^\d{4}$', word) and 1900 < int(word) < 2020):
        return "DATE"
    elif re.match(r'^\d+$', word):
        return "OTHER_NUM"
    elif re.match(r'^[A-Za-z]+\d+|\d+[A-Za-z]+', word):
        return "DIGIT_AND_ALPHA"
    elif word[0].isupper() and word[1:].islower():
        return "INITIAL_CAP"
    elif word.islower():
        return "LOWER_CASE"
    elif not word.isalnum():
        return "OTHER"
    else:
        return word


def process_and_get_pseudo_tagged_sets(tagged_training_set, tagged_test_set, frequency_threshold=5):
    """
    replaces rare training words and unknown test words with pseudo words and counts everything again
    """
    word_frequency = Counter(word for sentence in tagged_training_set for word, _ in sentence)
    low_freq_words = {word for word, freq in word_frequency.items() if freq < frequency_threshold}
    training_vocabulary = set(word_frequency.keys())
    unknown_words = set(word for sentence in tagged_test_set for word, _ in sentence if word not in training_vocabulary)

    def assign_pseudo_words(sentence, is_training=True):
        # the word changes to its pseudo word, the tag stays the real tag
        pseudo_sentence = []
        for word, tag in sentence:
            if word in low_freq_words or (not is_training and word in unknown_words):
                pseudo_sentence.append((create_psuedo_tag_for_word(word), tag))
            else:
                pseudo_sentence.append((word, tag))
        return pseudo_sentence

    pseudo_tagged_training_set = [assign_pseudo_words(sentence, True) for sentence in tagged_training_set]
    pseudo_tagged_test_set = [assign_pseudo_words(sentence, False) for sentence in tagged_test_set]

    word_tags_count, tag_counts = defaultdict(Counter), Counter()
    tag_transitions = defaultdict(lambda: defaultdict(int))

    for sentence in pseudo_tagged_training_set:
        for word, tag in sentence:
            word_tags_count[word][tag] += 1

    for sentence in pseudo_tagged_training_set:
        tags = ["START"] + [tag for (_, tag) in sentence] + ["STOP"]
        for i in range(len(tags) - 1):
            tag_counts[tags[i]] += 1
            tag_transitions[tags[i]][tags[i+1]] += 1

    pseudo_tag_emission_probabilities = defaultdict(dict)
    for word, tags in word_tags_count.items():
        for tag, count in tags.items():
            pseudo_tag_emission_probabilities[word][tag] = count / tag_counts[tag]

    pseudo_tag_transition_probabilities = {
        prev_tag: {next_tag: count / sum(transitions.values()) for next_tag, count in transitions.items()}
        for prev_tag, transitions in tag_transitions.items()
    }

    return (pseudo_tagged_training_set,
            pseudo_tagged_test_set,
            pseudo_tag_transition_probabilities,
            pseudo_tag_emission_probabilities,
            tag_counts,
            word_tags_count)


# e(ii) pseudo words with maximum likelihood
pseudo_tagged_training_set,\
 pseudo_tagged_test_set,\
 pseudo_tag_transition_probabilities,\
 pseudo_tag_emission_probabilities,\
 tag_counts,\
 word_tags_count\
    = process_and_get_pseudo_tagged_sets(tagged_training_set, tagged_test_set)

# known and unknown are still decided by the real words of the training set
results_without_laplace = test_viterbi_accuracy_with_known_unknown(pseudo_tagged_test_set, viterbi_algorithm,
                                                                   pseudo_tag_transition_probabilities,
                                                                   pseudo_tag_emission_probabilities,
                                                                   tag_counts,
                                                                   word_most_likely_tag,
                                                                   original_sentences=tagged_test_set)
print_results("HMM with pseudo words", results_without_laplace)

# e(iii) pseudo words with add one smoothing
pseudo_tag_emission_probabilities_with_laplace, pseudo_unseen_emission = \
    compute_emission_probabilities_with_laplace_smoothing(pseudo_tagged_training_set, tag_counts, word_tags_count)

results_with_laplace = test_viterbi_accuracy_with_known_unknown(pseudo_tagged_test_set, viterbi_algorithm,
                                                                pseudo_tag_transition_probabilities,
                                                                pseudo_tag_emission_probabilities_with_laplace,
                                                                tag_counts,
                                                                word_most_likely_tag,
                                                                pseudo_unseen_emission,
                                                                original_sentences=tagged_test_set)
print_results("HMM with pseudo words and add one smoothing", results_with_laplace)


def create_confusion_matrix(results):
    true_tags = [tag for sentence in tagged_test_set for _, tag in sentence]
    predicted_tags_final = [tag for predicted_tags in results[6] for tag in predicted_tags]

    unique_tags = sorted(set(true_tags + predicted_tags_final))
    tag_to_index = {tag: i for i, tag in enumerate(unique_tags)}

    conf_matrix = np.zeros((len(unique_tags), len(unique_tags)), dtype=int)
    for true, pred in zip(true_tags, predicted_tags_final):
        conf_matrix[tag_to_index[true], tag_to_index[pred]] += 1

    return conf_matrix, tag_to_index, unique_tags


def find_most_frequent_errors(conf_matrix, tag_to_index, unique_tags):
    errors = conf_matrix.copy()
    np.fill_diagonal(errors, 0)
    most_frequent_errors = {}

    for tag, index in tag_to_index.items():
        # tags with no mistakes are skipped
        if errors[index].max() == 0:
            continue
        most_frequent_errors[tag] = unique_tags[np.argmax(errors[index])]

    return most_frequent_errors


def print_top_errors(conf_matrix, unique_tags, n=10):
    errors = conf_matrix.copy()
    np.fill_diagonal(errors, 0)
    order = np.dstack(np.unravel_index(np.argsort(-errors, axis=None), errors.shape))[0][:n]
    for i, j in order:
        print(f"true {unique_tags[i]}, predicted {unique_tags[j]}: {errors[i, j]} times")


# confusion matrix for the pseudo words with add one smoothing
conf_matrix, tag_to_index, unique_tags = create_confusion_matrix(results_with_laplace)
most_frequent_errors = find_most_frequent_errors(conf_matrix, tag_to_index, unique_tags)

print()
print("Most frequent error of every tag (pseudo words with add one smoothing):")
for tag, error_tag in most_frequent_errors.items():
    print(f"{tag}: {error_tag}")

print()
print("The 10 most frequent errors:")
print_top_errors(conf_matrix, unique_tags)
