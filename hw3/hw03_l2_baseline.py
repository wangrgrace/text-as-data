"""
Name: Grace Wang (grw25)
Homework 3: L2 baseline logistic regression

This file is based on the Week 9 code (step11_train_classifier.py). It reads the train/test split that hw03_build_dataset.py writes to data/:
- train_core_vs_neg.json
- test_core_vs_neg.json
Each file is a list of [chunk_text, label] pairs (1 = CORE, 0 = NEG).

In addition to the Week 9 outputs, this file prints the number of non-zero coefficients, the top positive/negative words
(plus every non-zero word, if there are at most 50), and a sample of misclassified test chunks.

After hw03_build_dataset.py, run:
    python hw03_l2_baseline.py
"""

from pathlib import Path
import csv
import json
import textwrap

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

MODEL_NAME = "L2"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

TOP_N = 15             # top positive / negative words to print
N_ERRORS_TO_SHOW = 10  # misclassified chunks, split between false positives and negatives
MAX_NONZERO_TO_LIST = 50  # list every non-zero word if there are at most this many
RANDOM_SEED = 42


def print_top_words(title, indices, coefs, feature_names):
    print(f"\n{title}")
    for rank, i in enumerate(indices, start=1):
        print(f"  {rank:2d}. {feature_names[i]:<20} {coefs[i]:+.4f}")
    if len(indices) < TOP_N:
        print(f"  (only {len(indices)} non-zero coefficient(s) with this sign)")


# Load saved train/test split (same files as L1 model)
with open(DATA_DIR / "train_core_vs_neg.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(DATA_DIR / "test_core_vs_neg.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

X_train_texts = [t for (t, y) in train_data]
y_train = [y for (t, y) in train_data]

X_test_texts = [t for (t, y) in test_data]
y_test = np.array([y for (t, y) in test_data])

print(f"=== {MODEL_NAME} logistic regression ===")
print("Train size:", len(train_data))
print("Test size :", len(test_data))
print("Train labels:", np.bincount(y_train))
print("Test labels:", np.bincount(y_test))

# TF-IDF: same settings as Week 9 / the L1 model
vectorizer = TfidfVectorizer(
    lowercase=True,
    min_df=5,
    max_df=0.9
)

X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)
print("TF-IDF features:", X_train.shape[1])

# ---------------------------------------------------------------
# I. Model: Week 9 settings (LogisticRegression defaults to L2 penalty)
# ---------------------------------------------------------------
# The tutorial also passed n_jobs=1, but that was already the default,
# and on scikit-learn 1.8 or later it only raises a FutureWarning, so it's left out.
clf = LogisticRegression(
    max_iter=1000
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Evaluation: Week 9 metrics + number of non-zero coefficients (same as the L1 model)
print("\n1. Confusion matrix (rows = true, columns = predicted; 0 = NEG, 1 = CORE):")
print(confusion_matrix(y_test, y_pred))

print("\n2. Classification report:")
print(classification_report(y_test, y_pred, digits=3))

auc = roc_auc_score(y_test, y_prob)
print("3. ROC AUC:", round(auc, 4))

coefs = clf.coef_[0]
n_nonzero = np.count_nonzero(coefs)
print(f"\n4. Non-zero coefficients: {n_nonzero} of {coefs.size} ({n_nonzero / coefs.size:.1%})")

# ---------------------------------------------------------------
# III. "What the model learned"
# ---------------------------------------------------------------
feature_names = vectorizer.get_feature_names_out()
order = np.argsort(coefs)  # ascending: most negative first

# Keep only strictly positive and negative weights (same as the L1 model)
top_pos = [i for i in order[::-1][:TOP_N] if coefs[i] > 0]
top_neg = [i for i in order[:TOP_N] if coefs[i] < 0]

print_top_words(f"Top {TOP_N} positive-weight words (most predictive of CORE = 1):",
                top_pos, coefs, feature_names)
print_top_words(f"Top {TOP_N} negative-weight words (most predictive of NEG = 0):",
                top_neg, coefs, feature_names)

# List every non-zero word, if there are few enough to read (same as the L1 model)
nonzero = [i for i in order[::-1] if coefs[i] != 0]  # most positive first
if len(nonzero) <= MAX_NONZERO_TO_LIST:
    print(f"\nAll {len(nonzero)} non-zero coefficients (most positive to most negative):")
    for rank, i in enumerate(nonzero, start=1):
        print(f"  {rank:2d}. {feature_names[i]:<20} {coefs[i]:+.4f}")
else:
    print(f"\n{len(nonzero)} non-zero coefficients: too many to list, so only the top words are shown above.")

# Misclassified chunks (to examine weak-supervision / label-noise)
false_pos = np.where((y_test == 0) & (y_pred == 1))[0]
false_neg = np.where((y_test == 1) & (y_pred == 0))[0]
print(f"\nMisclassified test chunks: {len(false_pos)} false positives, "
      f"{len(false_neg)} false negatives")

# Split the 10 evenly
# If one error type has fewer than 5, take the rest from the other
n_fp = min(len(false_pos), max(N_ERRORS_TO_SHOW // 2, N_ERRORS_TO_SHOW - len(false_neg)))
n_fn = min(len(false_neg), N_ERRORS_TO_SHOW - n_fp)

rng = np.random.default_rng(RANDOM_SEED)
error_groups = [
    ("FALSE POSITIVES (label = NEG, model says CORE)", false_pos, n_fp),
    ("FALSE NEGATIVES (label = CORE, model says NEG)", false_neg, n_fn),
]
for title, idx, n_show in error_groups:
    sample = rng.choice(idx, size=n_show, replace=False)
    print(f"\n--- {title}: {len(sample)} of {len(idx)}, chosen at random ---")
    for i in sorted(sample):
        print(f"\n[test #{i}]  P(CORE) = {y_prob[i]:.3f}")
        print(textwrap.fill(X_test_texts[i], width=100,
                            initial_indent="    ", subsequent_indent="    "))

# Save misclassified chunks so we can compare the L1 and L2 errors by test index
OUTPUT_DIR.mkdir(exist_ok=True)
errors_path = OUTPUT_DIR / f"{MODEL_NAME.lower()}_misclassified.csv"
with open(errors_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["test_index", "true_label", "predicted_label", "p_core", "text"])
    for i in sorted(np.concatenate([false_pos, false_neg])):
        writer.writerow([i, y_test[i], y_pred[i], round(y_prob[i], 4), X_test_texts[i]])

print(f"\nSaved all {len(false_pos) + len(false_neg)} misclassified chunks to {errors_path}")
