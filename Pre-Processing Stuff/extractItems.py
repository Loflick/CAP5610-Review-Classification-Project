import json
import requests
from tqdm import tqdm

CATEGORIES = {
    "Electronics": {"target_reviews": 200_000, "target_items": 25_000},
    "Books": {"target_reviews": 200_000, "target_items": 25_000},
    "Clothing_Shoes_and_Jewelry": {
        "target_reviews": 200_000,
        "target_items": 25_000,
    },
    "Beauty_and_Personal_Care": {
        "target_reviews": 200_000,
        "target_items": 25_000,
    },
    "Home_and_Kitchen": {"target_reviews": 200_000, "target_items": 25_000},
}

ITEMS_OUTPUT = "asin_to_item.json"
REVIEWS_OUTPUT = "amazon_reviews_1M_sample.jsonl"


def run():
  asin_to_item = {}

  for category, config in CATEGORIES.items():
    target_items = config["target_items"]
    url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_{category}.jsonl"

    print(f"Fetching {target_items:,} items for {category}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    count = 0
    for line in response.iter_lines():
      if line:
        item = json.loads(line.decode("utf-8"))
        asin = item.get("parent_asin")

        if asin and asin not in asin_to_item:
          asin_to_item[asin] = {
              "title": item.get("title"),
              "price": item.get("price"),
              "category": category,
              "features": item.get("features", []),
              "average_rating": item.get("average_rating"),
          }
          count += 1

        if count >= target_items:
          break  

    print(f"Captured {count:,} items from {category} in seconds.")

  with open(ITEMS_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(asin_to_item, f, indent=2)
  print(f"\nSaved {len(asin_to_item):,} total items to '{ITEMS_OUTPUT}'.\n")

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
        if not line:
          continue

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


if __name__ == "__main__":
  run()