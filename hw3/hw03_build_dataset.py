"""
Name: Grace Wang (grw25)
Homework 3: Data prep
This script rebuilds the Week 9 train/test files from the raw texts (train_test_documents/*.txt).
It writes the train/test files in JSON format (data/train_core_vs_neg.json & data/test_core_vs_neg.json), where each chunk is a string and each label is an integer. 
The MAYBE chunks are saved separately (data/maybe_texts.json) because they are not used for training or testing.

Based on the tutorial for Weeks 8-9, the code:
- Segments each text into sentences
- Groups them into ~120-word chunks
- Keeps chunks with 5-200 tokens
- Weak-labels each chunk using tiered keywords (i.e., CORE = 1, MAYBE = 2, NEG = 0)
- Downsamples NEG to the size of CORE, shuffles, then splits 80/20 into train/test sets

Note that the script skips step 5 (Word2Vec) and does not save step 7's merchant_labeled_chunks.json, as neither changes the train/test files.
With the same NLTK version, its output files are identical to those from tutorial steps 7-9.

The files in data/ were built with NLTK 3.9.2. A different version may split sentences differently, which changes the train/test split.

Install packages nltk and gensim (pip install nltk gensim). Then, run:
    python hw03_build_dataset.py
"""

from pathlib import Path
import json
import random

import nltk
from gensim.utils import simple_preprocess

# Download required tokenizers
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Paths and chunking settings
BASE_DIR = Path(__file__).resolve().parent
TEXT_DIR = BASE_DIR / "train_test_documents"
DATA_DIR = BASE_DIR / "data"

TARGET_WORDS = 120  # group sentences into chunks of ~120 words
MIN_WORDS = 5
MAX_WORDS = 200

# Tier definitions
# Tier A: direct spellings / variants of the seed concept
TIER_A = {
    "merchant", "merchants",
    "marchant", "marchants"
}

# Tier B: closely related commercial roles / terms
TIER_B = {
    "factor", "chapman",
    "adventurer", "adventurers",
    "venturer", "venturers",
    "staple", "staplers",
    "trade", "purser"
}

# Tier C: "maybe" occupational neighborhood (often adjacent, not always merchant-specific)
TIER_C = {
    "clothier", "clothyer",
    "tailor", "tayler",
    "haberdasher",
    "goldsmith",
    "vintner",
    "brewer",
    "banker",
    "grazier",
    "jeweller"
}

def chunk_text(text, target_words=120):
    sentences = nltk.sent_tokenize(text)    # split text into sentences

    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        if not words:
            continue

        # If adding this sentence makes it exceed the target, finalize the chunk
        if current_len + len(words) > target_words and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

        current.append(sent)
        current_len += len(words)

    # Add any leftover sentences
    if current:
        chunks.append(" ".join(current))

    return chunks

# Chunk and label files
txt_paths = sorted(TEXT_DIR.glob("*.txt"))
labeled = []

print(f"nltk {nltk.__version__}")
print(f"Processing {len(txt_paths)} files...")

for n, path in enumerate(txt_paths, start=1):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunks = chunk_text(text, TARGET_WORDS)

    for c in chunks:
        tokens = simple_preprocess(c, deacc=True)
        n_tokens = len(tokens)

        # Filter by length
        if not (MIN_WORDS <= n_tokens <= MAX_WORDS):
            continue

        token_set = set(tokens)

        # Weak-label by tier
        if token_set & (TIER_A | TIER_B):
            label = 1   # CORE
        elif token_set & TIER_C:
            label = 2   # MAYBE
        else:
            label = 0   # NEG

        labeled.append((c, label))

    if n % 50 == 0 or n == len(txt_paths):
        print(f"  {n}/{len(txt_paths)} files, {len(labeled)} chunks so far")

print("Total chunks labeled:", len(labeled))

# Balance CORE vs. NEG
random.seed(42)  # use same seed as tutorial so sample and split are reproducible

core = [(t, 1) for (t, y) in labeled if y == 1]
neg = [(t, 0) for (t, y) in labeled if y == 0]
maybe = [t for (t, y) in labeled if y == 2]

print("Loaded:")
print("  CORE:", len(core))
print("  NEG :", len(neg))
print("  MAYBE:", len(maybe))

# NEG step
neg_sample = random.sample(neg, len(core))

training_data = core + neg_sample
random.shuffle(training_data)

print("Training set size (CORE + NEG):", len(training_data))

# Split data into train and test sets (80/20)
split = int(0.8 * len(training_data))
train_data = training_data[:split]
test_data = training_data[split:]

print("Train size:", len(train_data))
print("Test size :", len(test_data))
print("MAYBE size:", len(maybe))

DATA_DIR.mkdir(exist_ok=True)

# Save prepared datasets
with open(DATA_DIR / "train_core_vs_neg.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False)

with open(DATA_DIR / "test_core_vs_neg.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False)

with open(DATA_DIR / "maybe_texts.json", "w", encoding="utf-8") as f:
    json.dump(maybe, f, ensure_ascii=False)

print(f"Saved training, test, and MAYBE datasets to {DATA_DIR}")
