# Deep Learning & Product Review Classifcation
Deep Learning Approaches to Product Review Classification using MLP, CNN, LSTM, Transformers

Each model reads Amazon product reviews, embedded with GloVe, and predicts the item's rating class (0-4, meaning 1-5 stars).

- Dataset: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Embeddings: https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip

## Setup

### 1. GloVe vectors
Download `glove.840B.300d.zip` (link above, about 2 GB), extract `glove.840B.300d.txt` and put it in the `preprocessing/` folder. 

### 2. Generate the dataset
Run from the **repo root** (the output files are written to whatever folder you run from):
```
python preprocessing/gloveEmbedTest.py
```
This downloads items and reviews from Hugging Face, splits them, and embeds the reviews. It takes a long time and needs about 50 GB of free disk space in total.

What it does:
1. Picks up to 25,000 items from each of 5 categories (Electronics, Books, Clothing_Shoes_and_Jewelry, Beauty_and_Personal_Care, Home_and_Kitchen).
2. Labels each item with `rating` 0-4, its `average_rating` rounded and shifted down by 1.
3. Splits the **items** 80% / 10% / 10% into train / validation / test, stratified by rating with a fixed random seed. Every review belongs to the split of its item, so no item appears in more than one split.
4. Downloads the reviews for those items and writes them to the matching split.
5. GloVe-embeds each review into a `(n_words, 300)` matrix (first 100 words of title + text; words missing from GloVe become zeros).

To re-embed existing review files without downloading again, set `downloadNewFiles = False` in `main()`. Leaving it `True` empties and re-downloads everything.

### Generated files 
| Split | Items | Reviews | Embeddings |
|---|---|---|---|
| Train | `trainItems.json` | `trainReviews.json` | `trainEmbed.json` |
| Validation | `validationItems.json` | `validationReviews.json` | `validationEmbed.json` |
| Test | `testItems.json` | `testReviews.json` | `testEmbed.json` |

File formats, despite the `.json` extensions:
- `*Items.json`: pretty-printed JSON objects one after another (not a JSON array and not one object per line), with `title`, `category`, `rating`, `parent_asin`. Parse them with `json.JSONDecoder().raw_decode` in a loop.
- `*Reviews.json`: JSON lines, with `title`, `text`, `category`, `parent_asin`. There is no rating; look it up from the items file by `parent_asin`.
- `*Embed.json`: HDF5 files, laid out as `category / parent_asin / mat_<i>`, each dataset a float32 `(n_words, 300)` matrix.

Class balance is very skewed (train items: class 0 about 1%, 1 about 1.5%, 2 about 8%, 3 about 58%, 4 about 31%). Plan for this when training and when picking metrics.

## Using the embedded data in a model
Models live in `models/<name>/`. Put the repo root on the path, then import the helpers:
```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from preprocessing.readGloveTest import loadData
from preprocessing.gloveEmbedTest import CATEGORIES, TRAIN_EMBED_OUTPUT
```
(`parents[2]` is correct for a file at `models/<name>/file.py`. Adjust it if your file is somewhere else.)

Then stream one split, one category at a time:
```python
for category in CATEGORIES:
    for asin, matrices in loadData.stream_category(REPO_ROOT / TRAIN_EMBED_OUTPUT, category):
        ...
```
`matrices` is a list with one `(n_words, 300)` array per review of that item. The other split constants are `VALIDATION_EMBED_OUTPUT`, `TEST_EMBED_OUTPUT`, and the matching `*_ITEMS_OUTPUT` and `*_REVIEW_OUTPUT` names.

## Models
| Model | Folder |
|---|---|
| MLP | `models/MLP/` (`MLP.py` is the model, `runMLP.py` runs the pipeline) |
| Transformer | `models/transformer/` |
