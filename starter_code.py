import os
import nltk
from sklearn import metrics

# Download the NLTK POS tagger
nltk.download('averaged_perceptron_tagger_eng')


def evaluate(test_sentences: list[tuple[str, str]], tagged_test_sentences: list[tuple[str, str]]) -> None:
    """
    Evaluate the accuracy of our tagged test tokens against the ground-truth tags from PTB.

    args:
        test_sentences: list[tuple[str, str]]: The {tokens and "true" tags} pairs from the PTB test split
        tagged_test_sentences: list[tuple[str, str]]: The {tokens and predicted tags} pairs from our POS tagger under evaluation
    """
    # The "true" tags from PTB (y)
    gold = [str(tag) for sentence in test_sentences for token, tag in sentence]
    # The tags we predicted (y-hat)
    pred = [str(tag) for sentence in tagged_test_sentences for token, tag in sentence]
    # sklearn classification report
    print(metrics.classification_report(gold, pred, zero_division=0))

def get_token_tag_tuples(sent: str) -> list[tuple[str, str]]:
    """
    Process a sentence like:
    The/DT company/NN previously/RB reported/VBN  net/JJ of/IN  $/$ 2.3/CD million/CD ,/, or/CC  15/CD cents/NNS 
    Into an equivalent list of 2-tuples like:
    [('The', 'DT'), ('company', 'NN'), ('previously', 'RB'), ('reported', 'VBN'), ('net', 'JJ'), ('of', 'IN'), ('$', '$'), ('2.3', 'CD'), ('million', 'CD'), (',', ','), ('or', 'CC'), ('15', 'CD'), ('cents', 'NNS')]
    Where every element is a tuple containing a token and its tag.
    """
    return [nltk.tag.str2tuple(t) for t in sent.split()]

def get_tagged_sentences(text: str) -> list[str]:
    """
    Given a string of file PTB file contents that contains multiple sequences delimited by  "======================================" and newlines (it's a strange format, check it out)

    Return a sequence of sentences like:
    [' Richard/NNP W./NNP Lock/NNP ,/, retired/JJ  vice/NN president/NN and/CC  treasurer/NN of/IN  Owens-Illinois/NNP Inc./NNP ,/, was/VBD named/VBN  a/DT director/NN of/IN  this/DT transportation/NN industry/NN supplier/NN ,/, increasing/VBG  its/PRP$ board/NN to/TO  six/CD members/NNS ./. ']
    Where each element is a sentence where each token is tagged with its part of speech, using the token/tag format.
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

def main():
    # Set path for the directory containing the PTB dataset.
    datadir = os.path.join("data", "penn-treeban3-wsj", "wsj")

    # Load the training, dev, and test sets.
    train, dev, test = load_treebank_splits(datadir)

    ### An example of evaluation against the default NLTK POS tagger
    # Extract the (token, tag) tuples from the sentence of token/tag "words". These tags are the ground-truth tags from PTB.
    test_sentences: list[tuple[str, str]] = [get_token_tag_tuples(sent) for sent in test]  

    # Use NLTK's POS tagger to tag the sentence, using just the tokens from the previous (token, tag) tuples.
    tagged_test_sentences: list[tuple[str, str]] = [nltk.pos_tag([token for token, tag in sentence]) for sentence in test_sentences]

    # Evaluate the accuracy of the NLTK POS tagger against the ground-truth tags from PTB.
    evaluate(test_sentences, tagged_test_sentences)

if __name__ == "__main__":
    main()