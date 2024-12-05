'''Functions lifted from starter code: 
get_tagged_sentences
load_treebank_splits
'''

import os

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

def add_start_stop(data: list[str]) -> list[str]:
    """
    Add start and stop tokens to each sentence in the dataset.
    """
    new_data = []
    for sentence in data:
        new_data.append("<START> " + sentence + " <STOP>")
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

    # Print the first sentence in the training set.
    print(train[0])

if __name__ == "__main__":
    main()