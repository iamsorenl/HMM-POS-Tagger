'''Functions lifted from starter code: 
get_tagged_sentences
load_treebank_splits
'''

import os
from collections import defaultdict
import math

def viterbi(sentence, tags, transition_probs, emission_probs):
    """
    Viterbi algorithm for part-of-speech tagging.
    """
    # Create a path probability table: viterbi[state][time]
    viterbi_table = defaultdict(lambda: defaultdict(lambda: -math.inf)) # log-spce initialization to avoid underflow
    # create a backpointer table: backpointer[state][time]
    backpointer = defaultdict(dict)

    # Initialization
    for tag in tags:
        viterbi_table[tag][1] = math.log(transition_probs[("<START>", tag)]) + math.log(emission_probs.get((sentence[0], tag), 1e-10))
        backpointer[tag][1] = "<START>"

    # Recursion Step
    for t in range(2, len(sentence)):
        for tag_curr in tags:
            max_prob = -math.inf
            best_prev_tag = None
            for tag_prev in tags:
                prob = viterbi_table[tag_prev][t-1] + math.log(transition_probs[tag_prev, tag_curr]) + math.log(emission_probs[sentence[t-1], tag_curr])

                if prob > max_prob:
                    max_prob = prob
                    best_prev_tag = tag_prev
                
            viterbi_table[tag_curr][t] = max_prob
            backpointer[tag_curr][t] = best_prev_tag

    # Termination
    max_prob = -math.inf
    best_last_tag = None
    for tag in tags:
        prob = viterbi_table[tag][len(sentence) - 1] + math.log(transition_probs[tag, "<STOP>"])
        if prob > max_prob:
            max_prob = prob
            best_last_tag = tag

    # Backtrace
    best_path = [best_last_tag]
    for t in range(len(sentence) - 1, 1):
        best_path.insert(0, backpointer[best_path[0]][t])
    
    return best_path, max_prob


def compute_transition_probabilities(tag_bigrams, tag_counts, alpha, unique_tags):
    """
    Compute transition probabilities with add-alpha smoothing.
    """
    transition_probs = defaultdict(float)

    for t_prev in unique_tags:
        for t_curr in unique_tags:
            bigram_count = tag_bigrams[(t_prev, t_curr)]
            prev_tag_count = tag_counts[t_prev]

            # Compute smoothed probability
            transition_probs[(t_prev, t_curr)] = (bigram_count + alpha) / (prev_tag_count + alpha * len(unique_tags))

    return transition_probs

def compute_emission_probabilities(word_tag_counts, tag_counts, alpha, vocab):
    """
    Compute emission probabilities with add-alpha smoothing.
    """
    emission_probs = defaultdict(float)

    for (word, tag), count in word_tag_counts.items():
        tag_count = tag_counts[tag]

        # Compute smoothed probability
        emission_probs[(word, tag)] = (count + alpha) / (tag_count + alpha * len(vocab))

    return emission_probs

def calculate_counts(data: list[str]) -> tuple[defaultdict, defaultdict, defaultdict]:
    """
    Count tag bigrams and word-tag pairs in the dataset.
    """
    tag_bigrams = defaultdict(int)
    word_tag_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    for sentence in data:
        words_tags = sentence.split()
        for i in range(len(words_tags) - 1):
            word, tag = words_tags[i].rsplit("/", 1)
            next_tag = words_tags[i + 1].rsplit("/", 1)[1]

            # Update counts
            word_tag_counts[(word, tag)] += 1 # example: ('the', 'DT'): 1
            tag_counts[tag] += 1 # example: 'DT': 1
            tag_bigrams[(tag, next_tag)] += 1 # example: ('DT', 'NN'): 1

        # Update count for the final tag in the sentence
        final_tag = words_tags[-1].rsplit("/", 1)[1]
        tag_counts[final_tag] += 1

    return tag_bigrams, word_tag_counts, tag_counts

def get_tagged_sentences(text: str) -> list[str]:
    """
    Given a string of file PTB file contents that contains multiple sequences delimited by  "======================================" and newlines (it's a strange format, check it out)
    """
    sentences = []

    blocks = text.split("======================================")
    for block in blocks:
        sents = block.split("\n\n")
        for sent in sents:
            sent = sent.replace("\n", "").replace("[", "").replace("]", "")
            if sent != "":
                sentences.append(sent)

    return sentences

def load_treebank_splits(datadir: str) -> tuple[list[str], list[str], list[str]]:
    """
    The orchestrating function that returns a train, dev, and test sets given a string showing the filepath to the contents our downloaded subset of the Penn Treebank dataset.
    Each of these are lists of strings, where a string might look like:
    With/IN  stock/NN prices/NNS hovering/VBG near/IN  record/NN levels/NNS ,/,  a/DT number/NN of/IN  companies/NNS have/VBP been/VBN announcing/VBG  stock/NN splits/NNS ./. 
    """

    train = []
    dev = []
    test = []

    print("Loading treebank data...")

    for subdir, _, files in os.walk(datadir):
       for filename in files:
           if filename.endswith(".pos"):
               filepath = subdir + os.sep + filename
               with open(filepath, "r") as fh:
                   text = fh.read()
                   if int(subdir.split(os.sep)[-1]) in range(0, 19):
                       train += get_tagged_sentences(text)

                   if int(subdir.split(os.sep)[-1]) in range(19, 22):
                       dev += get_tagged_sentences(text)

                   if int(subdir.split(os.sep)[-1]) in range(22, 25):
                       test += get_tagged_sentences(text)

    return train, dev, test

def add_start_stop(data: list[str]) -> list[str]:
    """
    Add start and stop tokens to each sentence in the dataset.
    """
    new_data = []
    for sentence in data:
        new_data.append("<START>/<START> " + sentence + " <STOP>/<STOP>")
    return new_data

def main():
    # Set path for the directory containing the PTB dataset.
    datadir = os.path.join("data", "penn-treeban3-wsj", "wsj")

    # Load the training, dev, and test sets.
    train, dev, test = load_treebank_splits(datadir)

    # Add start and stop tokens to each sentence in the dataset.
    train = add_start_stop(train)
    dev = add_start_stop(dev)
    test = add_start_stop(test)

    # count tag bigrams and word-tag pairs in the dataset
    tag_bigrams, word_tag_counts, tag_counts = calculate_counts(train)

    # Extract unique tags and vocabulary
    unique_tags = list(tag_counts.keys())
    vocab = {word for (word, tag) in word_tag_counts.keys()}

    # Set smoothing parameter
    alpha = 1.0

    # Compute transition and emission probabilities
    transition_probs = compute_transition_probabilities(tag_bigrams, tag_counts, alpha, unique_tags)
    emission_probs = compute_emission_probabilities(word_tag_counts, tag_counts, alpha, vocab)

    # Viterbi algorithm


if __name__ == "__main__":
    main()