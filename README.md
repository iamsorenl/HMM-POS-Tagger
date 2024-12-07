---

# POS Tagging with Hidden Markov Models and Viterbi Algorithm

This project implements a **Hidden Markov Model (HMM)** for Part-of-Speech (POS) tagging using the **Viterbi Algorithm**. The implementation is designed to train on labeled data and decode the most probable sequence of POS tags for a given set of sentences.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Project Overview

POS tagging assigns grammatical tags (e.g., noun, verb, adjective) to words in a sentence. This project uses an **HMM** with the **Viterbi Algorithm** for decoding. Key features include:

- **HMM Model**:
  - Transition and emission probabilities with add-\( \alpha \) smoothing.
  - Log-space calculations for numerical stability.
- **Viterbi Algorithm**:
  - Efficient decoding with dynamic programming.
  - Implements backtracking to identify optimal tag sequences.

The model is trained on labeled text and evaluated on precision, recall, and F1 scores.

---

## Requirements

To set up the project, ensure you have the following dependencies installed:

- Python >= 3.11
- NumPy
- Pandas
- scikit-learn

### Installing Dependencies

Install the necessary packages using:

```bash
pip install -r requirements.txt
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/hmm-pos-tagger.git
cd hmm-pos-tagger
```

### 2. Create a Virtual Environment

To avoid package conflicts, it is recommended to use a virtual environment:

```bash
python3 -m venv venv
# Activate on MacOS/Linux
source venv/bin/activate
# Activate on Windows
venv\Scripts\activate
```

---

## Dataset Structure

Ensure the following files are present in the project directory:

- `train.csv`: Training dataset with sentences and POS tags.
- `test.csv`: Test dataset containing sentences for tagging.

### Example Data Format

**Training Dataset (`train.csv`):**

| ID  | Sentence                          | POS Tags                  |
|-----|-----------------------------------|---------------------------|
| 1   | The quick brown fox jumps         | DT JJ JJ NN VBZ           |

**Test Dataset (`test.csv`):**

| ID  | Sentence                          |
|-----|-----------------------------------|
| 1   | The lazy dog sleeps               |

---

## Usage

### Running the Training and Prediction Pipeline

To train the model and predict POS tags for test sentences, run the following command:

```bash
python hmm_tagger.py train.csv test.csv output.csv
```

This command will:

1. Train the HMM model using the training data.
2. Decode POS tags for the test data using the Viterbi algorithm.
3. Save the predictions to `output.csv`.

### Output Format

The output file `output.csv` will be structured as:

| ID  | POS Tags                   |
|-----|----------------------------|
| 1   | DT JJ NN VBZ               |

---

## Training the Model

The model uses an HMM framework with the following features:

- **Transition Probabilities**:
  - Captures the likelihood of transitioning from one POS tag to another.
- **Emission Probabilities**:
  - Captures the likelihood of a word being emitted by a specific tag.
- **Add-\( \alpha \) Smoothing**:
  - Addresses data sparsity by smoothing probabilities.

---

## Evaluation

### Metrics

The model's performance is evaluated using:

- **Token-level F1 Score**: Measures accuracy for individual tags.
- **Sequence-based Accuracy**: Measures how often the full sequence of predicted tags matches the ground truth.

### Example Results

| Metric              | Value  |
|---------------------|--------|
| Token-level F1      | 0.923  |
| Sequence Accuracy   | 0.871  |

---

## Hyperparameter Tuning

The smoothing parameter \( \alpha \) was tuned using grid search over values \([0.1, 0.5, 1.0, 2.0]\). The best performance was achieved with \( \alpha = 0.1 \).

---

## Limitations and Future Work

- **Unknown Words**:
  - Unseen words during training may affect decoding. A fallback strategy, such as assigning the most common tag, can be explored.
- **Dataset Size**:
  - Limited data may affect generalization to diverse contexts.
- **Future Directions**:
  - Extend the implementation to include higher-order HMMs or integrate neural network embeddings for improved word representations.

--- 
