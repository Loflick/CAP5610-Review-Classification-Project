import pandas as pd
import math

df_items = pd.read_json("asin_to_item.json", orient="index")
df_reviews = pd.read_json("amazon_reviews_1M_sample.jsonl", lines=True)

one = 0
two = 0
three = 0
four = 0
five = 0
for idx, row in df_reviews.iterrows():
    asin = row["title"]
    # val = math.ceil(row["average_rating"] - 0.5)
    print(asin)
    print(row["text"])
    break

    match val:
        case 1:
            one += 1
        case 2:
            two += 1
        case 3:
            three += 1
        case 4:
            four += 1
        case 5:
            five += 1
