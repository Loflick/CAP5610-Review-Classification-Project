import json
import requests
from tqdm import tqdm

CATEGORIES = {
    "Electronics": 200_000,
    "Books": 200_000,
    "Clothing_Shoes_and_Jewelry": 200_000,
    "Beauty_and_Personal_Care": 200_000,
    "Home_and_Kitchen": 200_000,
}

OUTPUT_FILE = "amazon_reviews_1M_sample.jsonl"


def stream_sample_reviews(categories, output_path):
  with open(output_path, "w", encoding="utf-8") as _:
    pass

  total_saved = 0

  for category, target_count in categories.items():
    url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{category}.jsonl"
    print(f"\n--- Streaming {target_count:,} reviews from '{category}' ---")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    count = 0
    with open(output_path, "a", encoding="utf-8") as f:
      for line in tqdm(response.iter_lines(), total=target_count, unit="lines"):
        if line:
          record = json.loads(line.decode("utf-8"))
          # Tag each record with its category domain for multi-task/ML tracking
          record["category"] = category

          f.write(json.dumps(record) + "\n")
          count += 1

          if count >= target_count:
            break

    total_saved += count
    print(f"Saved {count:,} reviews for {category}.")

  print(
      f"\nSuccess! Total of {total_saved:,} reviews written to '{output_path}'."
  )


if __name__ == "__main__":
  stream_sample_reviews(CATEGORIES, OUTPUT_FILE)