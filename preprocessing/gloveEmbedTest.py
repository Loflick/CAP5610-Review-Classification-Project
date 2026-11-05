import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import h5py
from collections import defaultdict
from tqdm import tqdm
import re
import requests
import hashlib
from sklearn.model_selection import train_test_split
import gc

CATEGORIES = {
    "Electronics": {"target_reviews": 200_000, "target_items": 25_000},
    "Books": {"target_reviews": 200_000, "target_items": 25_000},
    "Clothing_Shoes_and_Jewelry": {"target_reviews": 200_000, "target_items": 25_000,},
    "Beauty_and_Personal_Care": {"target_reviews": 200_000,"target_items": 25_000,},
    "Home_and_Kitchen": {"target_reviews": 200_000, "target_items": 25_000},
}

ITEMS_OUTPUT = "asin_to_item.json"
REVIEWS_OUTPUT = "amazon_reviews_1M_sample.jsonl"
MAX_REVIEWS_PER_ITEM = 30

TRAIN_ITEMS_OUTPUT = "trainItems.json"
VALIDATION_ITEMS_OUTPUT = "validationItems.json"
TEST_ITEMS_OUTPUT = "testItems.json"

TRAIN_REVIEW_OUTPUT = "trainReviews.json"
VALIDATION_REVIEW_OUTPUT = "validationReviews.json"
TEST_REVIEW_OUTPUT = "testReviews.json"

TRAIN_EMBED_OUTPUT = "trainEmbed.json"
VALIDATION_EMBED_OUTPUT = "validationEmbed.json"
TEST_EMBED_OUTPUT = "testEmbed.json"

class embedText :
    def load_glove_dict(glove_path, embedding_dim=300):
        glove_dict = {}
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().rsplit(" ", embedding_dim)
                word = parts[0]
                vector = np.asarray(parts[1:], dtype=np.float32)
                glove_dict[word] = vector
        return glove_dict

    def review_to_matrix(text, glove_dict, embedding_dim=300, max_words=100):
        words = re.findall(r"\b\w+\b", str(text).lower())
        words = words[:max_words]
        if not words:
            return np.zeros((1, embedding_dim), dtype=np.float32)

        matrix = np.array([
            glove_dict.get(w, np.zeros(embedding_dim, dtype=np.float32))
            for w in words
        ], dtype=np.float32)

        return matrix

    def embed_reviews(filePath, savePath):
        df_reviews = pd.read_json(filePath, lines=True)

        glove_path = './preprocessing/glove.840B.300d.txt'
        glove = embedText.load_glove_dict(glove_path, embedding_dim=300)

        test_review = None

        with h5py.File(savePath, "w") as f:
            for row in tqdm(df_reviews.itertuples(), total=len(df_reviews), desc="Processing"):
                category = str(row.category)
                asin = str(row.parent_asin)
                test_review = str(row.title) + " " + str(row.text)

                mat = embedText.review_to_matrix(test_review, glove, embedding_dim=300, max_words=100)

                cat_group = f.require_group(category)
                asin_group = cat_group.require_group(asin)
                mat_idx = len(asin_group)

                asin_group.create_dataset(f"mat_{mat_idx}", data=mat, compression="gzip")

class extractData:
    def extractValidItemsForCategory(category):
        target_items = CATEGORIES[category]["target_items"]
        url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_{category}.jsonl"

        response = requests.get(url, stream=True)
        response.raise_for_status()

        reviewCountLookup = defaultdict(int)
        itemCount = 0

        pbar = tqdm(total=target_items, unit="reviews")

        with open(ITEMS_OUTPUT, "a", encoding="utf-8") as out_r:
            for line in response.iter_lines(chunk_size=1024 * 1024):
                if not line: continue
                item = json.loads(line.decode("utf-8"))
                asin = item.get("parent_asin")

                pbar.update(1)

                avg_rating = item.get("average_rating")
                if avg_rating is None:
                    continue
                rating = max(0, min(4, round(float(avg_rating)) - 1))

                item_to_save = {
                    "title": item.get("title"),
                    "category": category,
                    "rating": rating,
                    "parent_asin": asin,
                }
                out_r.write(json.dumps(item_to_save) + "\n")
                itemCount += 1


                if itemCount > CATEGORIES[category]["target_items"]:
                    break
        pbar.close()

    def extractReviewsFromValidItems(asinLookup, category):
        target_reviews = CATEGORIES[category]["target_reviews"]
        url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{category}.jsonl"

        response = requests.get(url, stream=True)
        response.raise_for_status()

        pbar = tqdm(total=target_reviews, unit="reviews")

        count = 0

        with open(TRAIN_REVIEW_OUTPUT, "a", encoding="utf-8") as out_T, \
             open(VALIDATION_REVIEW_OUTPUT, "a", encoding="utf-8") as out_V, \
             open(TEST_REVIEW_OUTPUT, "a", encoding="utf-8") as out_Te:
                writers = {
                    "train": out_T,
                    "val": out_V,
                    "test": out_Te
                }
                for line in response.iter_lines(chunk_size=1024 * 1024):
                    if not line: continue

                    review = json.loads(line.decode("utf-8"))
                    asin = review.get("parent_asin")

                    split = asinLookup.get(asin)
                    target = writers.get(split)

                    if target:
                        review_to_save = {
                            "title": review.get("title"),
                            "text": review.get("text"),
                            "category": category,
                            "parent_asin": asin,
                        }
                        target.write(json.dumps(review_to_save) + "\n")
                        pbar.update(1)
                        count += 1

                    if count > target_reviews:
                        break

        pbar.close()

def main():
    with open(ITEMS_OUTPUT, "w", encoding="utf-8") as out_f:
        pass
    with open(TRAIN_ITEMS_OUTPUT, "w", encoding="utf-8") as out_f:
        pass
    with open(VALIDATION_ITEMS_OUTPUT, "w", encoding="utf-8") as out_f:
        pass
    with open(TEST_ITEMS_OUTPUT, "w", encoding="utf-8") as out_f:
        pass
    with open(TRAIN_REVIEW_OUTPUT, "w", encoding="utf-8") as out_f:
        pass
    with open(VALIDATION_REVIEW_OUTPUT, "w", encoding="utf-8") as out_f:
        pass
    with open(TEST_REVIEW_OUTPUT, "w", encoding="utf-8") as out_f:
        pass

    for category, config in CATEGORIES.items():    
        print(f"Starting pipeline for {category}")
        extractData.extractValidItemsForCategory(category)

    with open(ITEMS_OUTPUT, "r") as out_f:
        items = [json.loads(line) for line in out_f if line.strip()]

    df = pd.DataFrame(items)

    train_df, temp_df = train_test_split(df, test_size = 0.20, stratify=df["rating"], random_state=25269876)

    val_df, test_df = train_test_split(temp_df, test_size = 0.50, stratify=temp_df["rating"], random_state=25269876)

    train_df.to_json(TRAIN_ITEMS_OUTPUT, orient="records", indent=2, lines=True)
    val_df.to_json(VALIDATION_ITEMS_OUTPUT, orient="records", indent=2, lines=True)
    test_df.to_json(TEST_ITEMS_OUTPUT, orient="records", indent=2, lines=True)

    lookup = defaultdict(str)
    lookup = {asin: "train" for asin in train_df["parent_asin"]}
    lookup.update({asin: "val" for asin in val_df["parent_asin"]})
    lookup.update({asin: "test" for asin in test_df["parent_asin"]})

    del train_df
    del val_df
    del test_df

    gc.collect()

    for category, config in CATEGORIES.items():
        print("Downloading Reviews")
        extractData.extractReviewsFromValidItems(lookup, category)

    print("Extracting Finished")
    print("Starting embedding")
    embedText.embed_reviews(TRAIN_REVIEW_OUTPUT, TRAIN_EMBED_OUTPUT)
    embedText.embed_reviews(VALIDATION_REVIEW_OUTPUT, VALIDATION_EMBED_OUTPUT)
    embedText.embed_reviews(TEST_REVIEW_OUTPUT, TEST_EMBED_OUTPUT)


    print("All finished")

if __name__ == "__main__":
    main()