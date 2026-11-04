import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import sys
from pathlib import Path
import json
from tqdm import tqdm
import random
import torch.nn.functional as F
import hashlib
from sklearn.metrics import cohen_kappa_score, f1_score

import torch_directml
device = torch_directml.device()


sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing.readGloveTest import loadData
from preprocessing.gloveEmbedTest import CATEGORIES
from preprocessing.gloveEmbedTest import ITEMS_OUTPUT

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        norm_x = self.norm1(x)
        attn_output = self.self_attn(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attn_output)
        
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, d_model=300, num_heads=6, num_layers=4, d_ff=1024, max_seq_length=100, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src_mask = (src.abs().sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
        x = self.dropout(self.positional_encoding(src))

        for enc_layer in self.encoder_layers:
            x = enc_layer(x, src_mask)

        mask_expanded = (src.abs().sum(dim=-1) != 0).unsqueeze(-1)
        sum_vectors = (x * mask_expanded).sum(dim=1)
        token_counts = mask_expanded.sum(dim=1).clamp(min=1)
        pooled = sum_vectors / token_counts

        return self.fc(pooled)

class ReviewAttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, review_vectors, review_mask=None):
        # review_vectors: [B, K, d_model]
        # review_mask: [B, K] (True for real reviews, False for empty padding reviews)
        scores = self.scorer(review_vectors)  # [B, K, 1]
        
        if review_mask is not None:
            # Mask out dummy reviews so they don't get attention
            scores = scores.masked_fill(~review_mask.unsqueeze(-1), -1e9)
            
        weights = torch.softmax(scores, dim=1) # [B, K, 1]
        item_vector = (weights * review_vectors).sum(dim=1) # [B, d_model]
        return item_vector, weights

class HierarchicalProductClassifier(nn.Module):
    def __init__(self, num_classes, d_model=300, num_heads=6, num_layers=4, d_ff=1024, max_seq_length=100, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.review_attention = ReviewAttentionPooling(d_model)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_reviews):
        # batch_reviews: [B, K, max_len, d_model]
        B, K, max_len, d_model = batch_reviews.shape

        # 1. Identify which of the K reviews are real vs empty padding
        # Shape: [B, K]
        real_review_mask = (batch_reviews.abs().sum(dim=(-1, -2)) != 0)
        # Fallback if an item has 0 reviews: ensure at least review 0 is active
        real_review_mask[:, 0] = True 

        # 2. Flatten for Word-Level Encoder
        flat_reviews = batch_reviews.view(B * K, max_len, d_model)
        word_mask = (flat_reviews.abs().sum(dim=-1) != 0) # [B * K, max_len]

        # CRITICAL FIX FOR NaN:
        # If an entire review is padding, unmask the first token so softmax doesn't divide 0/0
        empty_reviews = ~word_mask.any(dim=-1, keepdim=True) # [B * K, 1]
        safe_mask = word_mask | empty_reviews                # [B * K, max_len]
        attn_mask = safe_mask.unsqueeze(1).unsqueeze(2)      # [B * K, 1, 1, max_len]

        x = self.dropout(self.pos_encoder(flat_reviews))
        for layer in self.encoder_layers:
            x = layer(x, attn_mask)

        # 3. Word Pooling
        mask_exp = safe_mask.unsqueeze(-1).float()
        token_counts = mask_exp.sum(dim=1).clamp(min=1.0)
        review_vectors = (x * mask_exp).sum(dim=1) / token_counts
        review_vectors = review_vectors.view(B, K, d_model)

        # 4. Review Pooling (Pass real_review_mask so dummy reviews receive 0 weight)
        item_vectors, weights = self.review_attention(review_vectors, review_mask=real_review_mask)

        # 5. Output Logits
        logits = self.fc(item_vectors)
        return logits, weights

def get_split(asin):
    h = int(hashlib.md5(asin.encode()).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    elif h < 90:
        return "val"
    else:
        return "test"

def compute_metrics(all_preds, all_targets):
    preds_t = all_preds.float()
    targets_t = all_targets.float()

    mae = torch.mean(torch.abs(preds_t - targets_t)).item()

    off_by_one = torch.mean((torch.abs(all_preds - all_targets) <= 1).float()).item()

    y_pred = all_preds.cpu().numpy()
    y_true = all_targets.cpu().numpy()

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return {
        "MAE": mae,
        "Off-by-one": off_by_one,
        "Macro F1": macro_f1,
        "QWK": qwk
    }

def main():
    labels = {}

    model = HierarchicalProductClassifier(
        num_classes=5,
        d_model=300,
        num_heads=6,
        num_layers=4,
        d_ff=1024,
        max_seq_length=100,
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, foreach=False)

    model.train()
    optimizer.zero_grad()

    batch_size = 16
    product_batch = []
    label_batch = []

    with open(ITEMS_OUTPUT, "r", encoding="utf-8") as out_f:
        for line in out_f:
            line = line.strip()
            if not line: continue

            data = json.loads(line)
            asin = data.get("parent_asin")
            if get_split(asin) != "train": continue
            rating = round(float(data.get("average_rating"))) - 1
            rating = max(0, min(4, rating)) # clamp between 0 and 4
            labels[asin] = rating
    print("Finished looking through items")

    count = 0

    counts = torch.bincount(torch.tensor(list(labels.values())), minlength=5).float()
    weights = 1.0 / (counts + 1e-5)
    weights = weights / weights.sum() * 5.0

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    pbar = tqdm(total = len(labels), unit="items")


    for category, config in CATEGORIES.items():
        for asin, matrices in loadData.stream_category("arrays.h5", category):
            if asin not in labels or len(matrices) == 0: continue

            reviews_to_use = random.sample(matrices, min(len(matrices), 20))

            prod_tensor = torch.zeros((20, 100, 300), dtype=torch.float32)
            for i, r in enumerate(reviews_to_use):
                l = min(len(r), 100)
                prod_tensor[i, :l] = torch.from_numpy(r[:l])

            product_batch.append(prod_tensor)
            label_batch.append(labels[asin])

            if len(product_batch) == batch_size:
                batch_x = torch.stack(product_batch).to(device)
                batch_y = torch.tensor(label_batch, dtype=torch.long, device=device)

                logits, _ = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                product_batch.clear()
                label_batch.clear()

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                pbar.update(batch_size)
        break

    torch.save(model.state_dict(), "transformer.pth")
    print("saved")

def testModel():
    labels = {}

    with open(ITEMS_OUTPUT, "r", encoding="utf-8") as out_f:
        for line in out_f:
            line = line.strip()
            if not line: continue

            data = json.loads(line)
            asin = data.get("parent_asin")
            if get_split(asin) != "val": continue
            rating = round(float(data.get("average_rating"))) - 1
            rating = max(0, min(4, rating)) # clamp between 0 and 4
            labels[asin] = rating
    print("Finished looking through items")

    model = HierarchicalProductClassifier(
        num_classes=5,
        d_model=300,
        num_heads=6,
        num_layers=4,
        d_ff=1024,
        max_seq_length=100,
        dropout=0.1
    ).to(device)
    model.load_state_dict(torch.load("transformer.pth", weights_only=False))
    model.eval()

    all_preds = []
    all_targets = []

    product_batch = []
    label_batch = []
    batch_size = 16
    max_len = 100

    with torch.no_grad():
        pbar = tqdm(total = 20000, unit="items")

        for category, config in CATEGORIES.items():
            for asin, matrices in loadData.stream_category("arrays.h5", category):
                if asin not in labels or len(matrices) == 0: continue

                reviews_to_use = random.sample(matrices, min(len(matrices), 20))

                prod_tensor = torch.zeros((20, max_len, 300), dtype=torch.float32)
                for i, r in enumerate(reviews_to_use):
                    l = min(len(r), 100)
                    prod_tensor[i, :l] = torch.from_numpy(r[:l])

                product_batch.append(prod_tensor)
                label_batch.append(labels[asin])

                if len(product_batch) == batch_size:
                    batch_x = torch.stack(product_batch).to(device)

                    logits, _ = model(batch_x)
                    preds = torch.argmax(logits, dim=1)

                    all_preds.extend(preds.cpu().tolist())
                    all_targets.extend(label_batch)

                    product_batch.clear()
                    label_batch.clear()

                    pbar.update(batch_size)
        if len(product_batch) > 0:
            batch_x = torch.stack(product_batch).to(device)
            logits, _ = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(label_batch)
    all_preds_tensor = torch.tensor(all_preds)
    all_targets_tensor = torch.tensor(all_targets)

    metrics = compute_metrics(all_preds_tensor, all_targets_tensor)

    print("\n" + "="*35)
    print("        FINAL TEST RESULTS        ")
    print("="*35)
    print(f"Total Test Items:        {len(all_preds)}")
    print(f"Mean Absolute Error:     {metrics['MAE']:.4f}")
    print(f"Off-by-1 Accuracy:       {metrics['Off-by-one'] * 100:.2f}%")
    print(f"Macro-Averaged F1:       {metrics['Macro F1']:.4f}")
    print(f"Quadratic Weighted Kappa:{metrics['QWK']:.4f}")
    print("="*35)



    
if __name__ == "__main__":
    print("Training Model on first category")
    main()
    print("Testing model on first category")
    testModel()