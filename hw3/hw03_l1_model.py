"""
Name: Grace Wang (grw25)
Homework 3: L1-regularized logistic regression

This file is based on the Week 9 code. It reads the train/test split that hw03_build_dataset.py writes to data/:
- train_core_vs_neg.json
- test_core_vs_neg.json
Each file is a list of [chunk_text, label] pairs (1 = CORE, 0 = NEG).
    
This file is identical to hw03_l2_baseline.py in all aspects (JSON split, TF-IDF settings, outputs) except for the model: 
an L1 penalty with the liblinear solver, max_iter=2000 as in the assignment's example, and a fixed random_state.

After hw03_build_dataset.py, run:
    python hw03_l1_model.py
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

MODEL_NAME = "L1"
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


# Load saved train/test split (same files as L2 baseline)
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

# TF-IDF: same settings as Week 9 / the L2 baseline
vectorizer = TfidfVectorizer(
    lowercase=True,
    min_df=5,
    max_df=0.9
)

X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)
print("TF-IDF features:", X_train.shape[1])

# ---------------------------------------------------------------
# II. Model: L1 penalty + liblinear solver
# ---------------------------------------------------------------
# scikit-learn 1.7.2; on 1.8 or later, use l1_ratio=1 instead (see footnote 1 from Homework_3.pdf)
# liblinear shuffles the training data on each fit, so a fixed random_state 
# is necessary to ensure the coefficients remain consistent from run to run.
clf = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    max_iter=2000,
    random_state=RANDOM_SEED
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Evaluation: Week 9 metrics + number of non-zero coefficients (same as the L2 baseline)
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

# L1 sets most weights to exactly 0, so keep only strictly positive and negative ones
top_pos = [i for i in order[::-1][:TOP_N] if coefs[i] > 0]
top_neg = [i for i in order[:TOP_N] if coefs[i] < 0]

print_top_words(f"Top {TOP_N} positive-weight words (most predictive of CORE = 1):",
                top_pos, coefs, feature_names)
print_top_words(f"Top {TOP_N} negative-weight words (most predictive of NEG = 0):",
                top_neg, coefs, feature_names)

# List every non-zero word, if there are few enough to read (same as the L2 baseline)
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
