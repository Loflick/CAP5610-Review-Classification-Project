"""Runs the MLP pipeline: loads the pre-split data, trains, validate, tests."""

import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score
from tensorflow.keras.utils import set_random_seed
from tqdm import tqdm
from MLP import DEFAULT_CONFIG, EMBED_DIM, MLP, NUM_CLASSES

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from preprocessing.readGloveTest import loadData
from preprocessing.gloveEmbedTest import (
    CATEGORIES,
    TRAIN_ITEMS_OUTPUT,
    VALIDATION_ITEMS_OUTPUT,
    TEST_ITEMS_OUTPUT,
    TRAIN_EMBED_OUTPUT,
    VALIDATION_EMBED_OUTPUT,
    TEST_EMBED_OUTPUT,
)

SEED = 42
CLASSES = list(range(NUM_CLASSES))

# Data splits: (items.json, embeddings.npz)
SPLITS = {
    "train": (ROOT / TRAIN_ITEMS_OUTPUT, ROOT / TRAIN_EMBED_OUTPUT),
    "validation": (ROOT / VALIDATION_ITEMS_OUTPUT, ROOT / VALIDATION_EMBED_OUTPUT),
    "test": (ROOT / TEST_ITEMS_OUTPUT, ROOT / TEST_EMBED_OUTPUT),
}


def load_labels(items_path):
    """Maps each item to its rating."""
    text = Path(items_path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()

    labels = {}
    pos = 0

    # Parses JSON objects
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break

        item, pos = decoder.raw_decode(text, pos)
        labels[item["parent_asin"]] = item["rating"]

    return labels


def load_split(split):
    """Average each item's word embeddings into one vector."""
    items_file, embed_file = SPLITS[split]
    cache = ROOT / f"mlp_pooled_{split}.npz"

    # Init data from cache
    if cache.exists():
        data = np.load(cache)
        return data["X"], data["y"]

    # Init item ratings
    labels = load_labels(items_file)
    x, y = [], []

    # Pool each items reviews
    for category in CATEGORIES:
        items = loadData.stream_category(embed_file, category)

        # Pool each item's reviews into one vector
        for asin, reviews in tqdm(items, desc=f"{split} {category}", unit="item"):
            if asin not in labels or len(reviews) == 0:
                continue

            total = np.zeros(EMBED_DIM, dtype=np.float32)
            words = 0

            # Pool all reviews
            for review in reviews:
                total += review.sum(axis=0)
                words += len(review)

            # x = (1 / N) * sum(e_i), e_i 
            x.append(total / words)
            y.append(labels[asin])

    # Convert to numpy arrs
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    # Save to cache
    np.savez(cache, X=x, y=y)

    return x, y


def class_weights(ratings, power=1.0):
    """Give rarer classes larger loss weights, so they are not ignored."""
    # Count items per class
    counts = np.bincount(np.asarray(ratings), minlength=NUM_CLASSES)
    counts = counts.astype(np.float32)

    # Rarer classes have higher weight
    # w_c = (1 / n_c)^p
    weights = (1.0 / (counts + 1e-5)) ** power

    # Normalize: sum(w_c) = C
    weights = weights / weights.sum() * NUM_CLASSES

    result = {}

    # Maps class id to weight
    for i, weight in enumerate(weights):
        result[i] = float(weight)

    return result


def compute_metrics(y_true, y_pred):
    """Compute QWK, macro-F1, MAE, and off-by-1 accuracy."""
    # Error per item: |y - y_hat|
    error = np.abs(y_true - y_pred)

    # QWK = 1 - sum(w_ij * O_ij) / sum(w_ij * E_ij), w_ij = (i - j)^2
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=CLASSES)

    # Macro-F1 = (1 / C) * sum(F1_c), F1_c = 2PR / (P + R)
    f1 = f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)

    # MAE = (1 / n) * sum(|y - y_hat|)
    mae = float(error.mean())

    # Off-by-1 = (1 / n) * count(|y - y_hat| <= 1)
    off_by_1 = float((error <= 1).mean())

    return {
        "qwk": qwk,
        "macro_f1": f1,
        "mae": mae,
        "off_by_1_accuracy": off_by_1,
    }


def print_results(y_true, y_pred):
    """Print the final test results."""
    metrics = compute_metrics(y_true, y_pred)

    print("\n" + "=" * 35)
    print("        FINAL TEST RESULTS        ")
    print("=" * 35)
    print(f"Total Test Items:        {len(y_pred)}")
    print(f"Mean Absolute Error:     {metrics['mae']:.4f}")
    print(f"Off-by-1 Accuracy:       {metrics['off_by_1_accuracy'] * 100:.2f}%")
    print(f"Macro-Averaged F1:       {metrics['macro_f1']:.4f}")
    print(f"Quadratic Weighted Kappa:{metrics['qwk']:.4f}")
    print("=" * 35)


def main(config=DEFAULT_CONFIG):
    """Train the MLP and report results."""
    # Fix seed
    set_random_seed(SEED)

    # Init train, val, test
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("validation")
    x_test, y_test = load_split("test")

    # Compute class weights
    ratings = list(load_labels(SPLITS["train"][0]).values())
    weights = class_weights(ratings, config["weight_power"])

    # Build MLP
    model = MLP(
        hidden_sizes=config["hidden_sizes"],
        activation=config["activation"],
        learning_rate=config["learning_rate"],
    )
    # Fit MLP
    model.fit(
        x_train, y_train,
        x_val, y_val,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        class_weight=weights,
    )

    # Run on val set
    val_pred = model.predict(x_val)
    print("val", compute_metrics(y_val, val_pred))

    # Run test set
    test_pred = model.predict(x_test)
    print_results(y_test, test_pred)


if __name__ == "__main__":
    main()
