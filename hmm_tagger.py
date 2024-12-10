'''Functions lifted from starter code: 
get_tagged_sentences
load_treebank_splits
'''

import os
from collections import defaultdict, Counter
import math

def viterbi(sentence, tags, transition_probs, emission_probs):
    """
    Viterbi algorithm for part-of-speech tagging.
    """
    # Create a path probability table: viterbi[state][time]
    viterbi_table = defaultdict(lambda: defaultdict(lambda: -math.inf))  # Log-space initialization to avoid underflow
    # Create a backpointer table: backpointer[state][time]
    backpointer = defaultdict(dict)

    # Initialization
    #print("Starting Initialization Step...")
    for tag in tags:
        transition_prob = transition_probs.get(("<START>", tag), math.log(1e-10))
        emission_prob = emission_probs.get((sentence[0], tag), math.log(1e-10))
        viterbi_table[tag][1] = transition_prob + emission_prob
        backpointer[tag][1] = "<START>"
        #print(f"Init {tag}: Transition={transition_prob}, Emission={emission_prob}, Total={viterbi_table[tag][1]}")

    # Recursion Step
    #print("Starting Recursion Step...")
    for t in range(2, len(sentence) + 1):
        for tag_curr in tags:
            max_prob = -math.inf
            best_prev_tag = None
            for tag_prev in tags:
                transition_prob = transition_probs.get((tag_prev, tag_curr), math.log(1e-10))
                emission_prob = emission_probs.get((sentence[t-1], tag_curr), math.log(1e-10))
                prob = viterbi_table[tag_prev][t-1] + transition_prob + emission_prob
                #print(f"Step {t}, From {tag_prev} to {tag_curr}: Transition={transition_prob}, Emission={emission_prob}, Total={prob}")
                if prob > max_prob:
                    max_prob = prob
                    best_prev_tag = tag_prev
            viterbi_table[tag_curr][t] = max_prob
            backpointer[tag_curr][t] = best_prev_tag

    # Termination
    #print("Starting Termination Step...")
    max_prob = -math.inf
    best_last_tag = None
    for tag in tags:
        transition_prob = transition_probs.get((tag, "<STOP>"), math.log(1e-10))
        prob = viterbi_table[tag][len(sentence)] + transition_prob
        #print(f"Termination {tag}: Transition={transition_prob}, Total={prob}")
        if prob > max_prob:
            max_prob = prob
            best_last_tag = tag

    # Backtrace
    best_path = [best_last_tag]
    for t in range(len(sentence), 1, -1):
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
            word_tag_counts[(word, tag)] += 1
            tag_counts[tag] += 1
            tag_bigrams[(tag, next_tag)] += 1

        # Update count for the final tag in the sentence
        final_tag = words_tags[-1].rsplit("/", 1)[1]
        tag_counts[final_tag] += 1

    return tag_bigrams, word_tag_counts, tag_counts

def get_tagged_sentences(text: str) -> list[str]:
    """
    Parse PTB file contents into sentences with POS tags.
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
    Load train, dev, and test splits of the dataset.
    """
    train, dev, test = [], [], []

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
    Add start and stop tokens to each sentence.
    """
    return [f"<START>/<START> {sentence} <STOP>/<STOP>" for sentence in data]

def baseline_model(train, test):
    """
    Baseline model that assigns the most frequent tag for each word.
    """
    word_tag_freq = defaultdict(lambda: defaultdict(int))
    for sentence in train:
        words_tags = sentence.split()
        for wt in words_tags:
            word, tag = wt.rsplit("/", 1)
            word_tag_freq[word][tag] += 1

    # Determine the most frequent tag for each word
    baseline_tags = {word: max(tags, key=tags.get) for word, tags in word_tag_freq.items()}

    correct, total = 0, 0
    for sentence in test:
        words_tags = sentence.split()
        words = [wt.rsplit("/", 1)[0] for wt in words_tags[1:-1]]
        true_tags = [wt.rsplit("/", 1)[1] for wt in words_tags[1:-1]]

        # Predict tags using the baseline model
        predicted_tags = [baseline_tags.get(word, "NN") for word in words]
        correct += sum(p == t for p, t in zip(predicted_tags, true_tags))
        total += len(true_tags)

    return correct / total

def debug_single_example():
    """
    Debug using the example from the assignment PDF.
    """
    # Words and tags from the PDF example
    words = ["example_word1", "example_word2"]
    true_tags = ["VB", "NN"]

    # Hardcoded transition probabilities
    transition_probs = {
        ("<START>", "VB"): 4,
        ("VB", "NN"): 9,
        ("NN", "<STOP>"): 1,
    }
    # Hardcoded emission probabilities
    emission_probs = {
        ("example_word1", "VB"): 1,
        ("example_word2", "NN"): 1,
    }

    # Convert probabilities to log space
    transition_probs = {k: math.log(v) for k, v in transition_probs.items()}
    emission_probs = {k: math.log(v) for k, v in emission_probs.items()}

    # Default missing probabilities to math.log(1e-10)
    for k in [("<START>", "VB"), ("VB", "NN"), ("NN", "<STOP>")]:
        if k not in transition_probs:
            transition_probs[k] = math.log(1e-10)
    for k in [("example_word1", "VB"), ("example_word2", "NN")]:
        if k not in emission_probs:
            emission_probs[k] = math.log(1e-10)

    # Calculate the gold sequence score
    gold_score = (
        transition_probs[("<START>", true_tags[0])] +
        emission_probs[(words[0], true_tags[0])] +
        transition_probs[(true_tags[0], true_tags[1])] +
        emission_probs[(words[1], true_tags[1])] +
        transition_probs[(true_tags[1], "<STOP>")]
    )

    # Run Viterbi to get predicted tags and score
    predicted_tags, predicted_score = viterbi(words, ["VB", "NN"], transition_probs, emission_probs)

    # Output the results
    print(f"Sentence: {' '.join(words)}")
    print(f"True Tags: {true_tags}")
    print(f"Predicted Tags: {predicted_tags}")
    print(f"Gold Sequence Score: {gold_score}")
    print(f"Viterbi Predicted Score: {predicted_score}")

    # Validate the outputs
    if not math.isclose(gold_score, predicted_score, rel_tol=1e-9):
        print("Mismatch detected!")
        print("Gold Score Calculation Breakdown:")
        print(f"    Transition (<START> -> {true_tags[0]}): {transition_probs[('<START>', true_tags[0])]}")
        print(f"    Emission ({words[0]} -> {true_tags[0]}): {emission_probs[(words[0], true_tags[0])]}")
        print(f"    Transition ({true_tags[0]} -> {true_tags[1]}): {transition_probs[(true_tags[0], true_tags[1])]}")
        print(f"    Emission ({words[1]} -> {true_tags[1]}): {emission_probs[(words[1], true_tags[1])]}")
        print(f"    Transition ({true_tags[1]} -> <STOP>): {transition_probs[(true_tags[1], '<STOP>')]}")
    assert math.isclose(gold_score, predicted_score, rel_tol=1e-9), \
        "Viterbi failed to find the highest scoring sequence!"
    assert predicted_tags == true_tags, \
        "Viterbi predicted the wrong sequence!"

    print("Debugging successful: Viterbi finds the correct sequence with the correct score.")

def main():
    datadir = os.path.join("data", "penn-treeban3-wsj", "wsj")
    train, dev, test = load_treebank_splits(datadir)
    train, dev, test = map(add_start_stop, [train, dev, test])
    
    # Debugging using the example from the assignment PDF
    debug_single_example()

    # Calculate counts for the training data
    tag_bigrams, word_tag_counts, tag_counts = calculate_counts(train)
    unique_tags = list(tag_counts.keys())
    vocab = {word for (word, _) in word_tag_counts.keys()}

    # Hyperparameter tuning for alpha
    best_alpha, best_accuracy = 0, 0
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        transition_probs = compute_transition_probabilities(tag_bigrams, tag_counts, alpha, unique_tags)
        emission_probs = compute_emission_probabilities(word_tag_counts, tag_counts, alpha, vocab)
        correct, total = 0, 0
        for sentence in dev:
            words_tags = sentence.split()
            words = [wt.rsplit("/", 1)[0] for wt in words_tags[1:-1]]
            true_tags = [wt.rsplit("/", 1)[1] for wt in words_tags[1:-1]]
            predicted_tags, _ = viterbi(words, unique_tags, transition_probs, emission_probs)
            correct += sum(p == t for p, t in zip(predicted_tags, true_tags))
            total += len(true_tags)
        accuracy = correct / total
        if accuracy > best_accuracy:
            best_alpha, best_accuracy = alpha, accuracy

    # Add this line after the loop
    print(f"Best alpha: {best_alpha} with accuracy: {best_accuracy:.2%}")

    # Compute final transition and emission probabilities with best alpha
    transition_probs = compute_transition_probabilities(tag_bigrams, tag_counts, best_alpha, unique_tags)
    emission_probs = compute_emission_probabilities(word_tag_counts, tag_counts, best_alpha, vocab)

    # Evaluate baseline model
    baseline_accuracy = baseline_model(train, test)

    # Evaluate HMM model on test set
    correct, total = 0, 0
    confusion = Counter()
    for sentence in test:
        words_tags = sentence.split()
        words = [wt.rsplit("/", 1)[0] for wt in words_tags[1:-1]]
        true_tags = [wt.rsplit("/", 1)[1] for wt in words_tags[1:-1]]
        predicted_tags, _ = viterbi(words, unique_tags, transition_probs, emission_probs)
        correct += sum(p == t for p, t in zip(predicted_tags, true_tags))
        total += len(true_tags)
        confusion.update((t, p) for t, p in zip(true_tags, predicted_tags))


    # Calculate precision, recall, and F1 score
    precision = {}
    recall = {}
    f1 = {}

    for tag in unique_tags:
        precision_denominator = sum(confusion[(t, tag)] for t in unique_tags)
        recall_denominator = sum(confusion[(tag, p)] for p in unique_tags)
        
        if precision_denominator > 0:
            precision[tag] = confusion[(tag, tag)] / precision_denominator
        else:
            precision[tag] = 0.0  # No predictions for this tag

        if recall_denominator > 0:
            recall[tag] = confusion[(tag, tag)] / recall_denominator
        else:
            recall[tag] = 0.0  # No true labels for this tag

        if precision[tag] + recall[tag] > 0:
            f1[tag] = 2 * precision[tag] * recall[tag] / (precision[tag] + recall[tag])
        else:
            f1[tag] = 0.0  # Precision and recall are both zero
        macro_precision = sum(precision.values()) / len(unique_tags)
        macro_recall = sum(recall.values()) / len(unique_tags)
        macro_f1 = sum(f1.values()) / len(unique_tags)

    print(f"Test Set Accuracy: {correct / total:.2%}")
    print(f"Baseline Accuracy: {baseline_accuracy:.2%}")
    print(f"Macro Precision: {macro_precision:.2%}, Macro Recall: {macro_recall:.2%}, Macro F1: {macro_f1:.2%}")

    # Output the full confusion matrix to a file
    with open("confusion_matrix.txt", "w") as f:
        f.write("True Tag\tPredicted Tag\tCount\n")
        for (true, pred), count in confusion.items():
            f.write(f"{true}\t{pred}\t{count}\n")


if __name__ == "__main__":
    main()
