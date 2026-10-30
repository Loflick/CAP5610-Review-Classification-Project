import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def embedding_for_vocab(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  
    embedding_matrix_vocab = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix_vocab[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix_vocab

df_reviews = pd.read_json("amazon_reviews_1M_sample.jsonl", lines=True)

testReview = ""

uniqueWords = []
uniqueWordsCount = {}

test_review = None
for idx, row in df_reviews.iterrows():
    test_review = row
    testReview += row["title"] + " "
    testReview += row["text"]
    break

tokenizer = Tokenizer(num_words=len(testReview), oov_token="<OOV>")
tokenizer.fit_on_texts(testReview)
sequences = tokenizer.texts_to_sequences(testReview)

embedding_dim = 300
glove_path = './glove.840B.300d.txt'

embedding_matrix_vocab = embedding_for_vocab(glove_path, tokenizer.word_index, embedding_dim)
first_word_index = 1  
print("Dense vector for word with index 1 =>", embedding_matrix_vocab[0])

save = {
    test_review["parent_asin"]: embedding_matrix_vocab,
}

with open("arrays.json", "w") as f:
    json.dump(save, f, indent=4)