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

CATEGORIES = {
    "Electronics": {"target_reviews": 200_000, "target_items": 25_000},
    "Books": {"target_reviews": 200_000, "target_items": 25_000},
    "Clothing_Shoes_and_Jewelry": {"target_reviews": 200_000, "target_items": 25_000,},
    "Beauty_and_Personal_Care": {"target_reviews": 200_000,"target_items": 25_000,},
    "Home_and_Kitchen": {"target_reviews": 200_000, "target_items": 25_000},
}

ITEMS_OUTPUT = "asin_to_item.json"
REVIEWS_OUTPUT = "amazon_reviews_1M_sample.jsonl"

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

    def embed_reviews():
        df_reviews = pd.read_json("amazon_reviews_1M_sample.jsonl", lines=True)

        glove_path = './Pre-Processing Stuff\glove.840B.300d.txt'
        glove = embedText.load_glove_dict(glove_path, embedding_dim=300)

        test_review = None

        with h5py.File("arrays.h5", "w") as f:
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
    def extractItems():
        asin_to_item = {}

        for category, config in CATEGORIES.items():
            target_items = config["target_items"]
            url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_{category}.jsonl"

            print(f"Fetching {target_items:,} items for {category}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            count = 0
            for line in response.iter_lines(chunk_size=1024 * 1024):
                if line:
                    item = json.loads(line.decode("utf-8"))
                    asin = item.get("parent_asin")

                    if asin and asin not in asin_to_item:
                        if (item.get("rating_number") or 0) < 10: continue
                        asin_to_item[asin] = {"title": item.get("title"), "price": item.get("price"), "category": category, "features": item.get("features", []), "average_rating": item.get("average_rating"),}
                        count += 1

                    if count >= target_items:
                        break  

            print(f"Captured {count:,} items from {category} in seconds.")

        with open(ITEMS_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(asin_to_item, f, indent=2)
        print(f"\nSaved {len(asin_to_item):,} total items to '{ITEMS_OUTPUT}'.\n")
        return asin_to_item


    def extractReviews(categories, asin_to_item):
        with open(REVIEWS_OUTPUT, "w", encoding="utf-8") as out_f:
            pass  

        for category, config in CATEGORIES.items():
            target_reviews = config["target_reviews"]
            url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{category}.jsonl"

            print(f"Streaming {target_reviews:,} reviews for {category}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            saved_reviews = 0
            pbar = tqdm(total=target_reviews, unit="reviews")

            with open(REVIEWS_OUTPUT, "a", encoding="utf-8") as out_f:
                for line in response.iter_lines():
                    if not line: continue

                    review = json.loads(line.decode("utf-8"))
                    asin = review.get("parent_asin")

                    if asin in asin_to_item:
                        review["category"] = category
                        out_f.write(json.dumps(review) + "\n")
                        saved_reviews += 1
                        pbar.update(1)

                        if saved_reviews >= target_reviews:
                            break

            pbar.close()

def main():
    print("Extracting Items")
    itemsTaken = extractData.extractItems()
    print("Item Data Downloaded")

    print()
    print("Extracting reviews relative to found Items")
    extractData.extractReviews(CATEGORIES, itemsTaken)
    print("Reviews extracted")
    print()

    print("Starting review embed")
    embedText.embed_reviews()
    print("Finished")

if __name__ == "__main__":
    main()