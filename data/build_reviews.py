"""Build `reviews.csv` from the Amazon Polarity dataset.

Amazon Polarity holds 4 million product reviews labeled positive or negative, which is
far more than a teaching notebook needs and 1.1 GB on disk. This script takes the test
split alone, keeps the short reviews, and samples a small balanced set.

Short reviews are the point. A bag of words needs words to overlap between the training
set and a new document, so it does badly on texts of a couple of dozen words, which is
exactly where an embedding model earns its keep. Long reviews repeat themselves enough
that TF-IDF catches up, and the lesson disappears.

Run from this directory:

    uv run --with pyarrow python build_reviews.py
"""

import pandas as pd
from huggingface_hub import hf_hub_download

# One 117 MB file from the test split; the four training shards are not needed.
PARQUET = "amazon_polarity/test-00000-of-00001.parquet"
MIN_WORDS, MAX_WORDS = 8, 40
PER_CLASS = 1500
SEED = 42

path = hf_hub_download("fancyzhx/amazon_polarity", PARQUET, repo_type="dataset")
reviews = pd.read_parquet(path)

# `content` is the review body; `title` is its headline and is not used, because a
# headline such as "Terrible!" gives the label away.
reviews["text"] = reviews["content"].str.strip()
word_count = reviews["text"].str.split().str.len()
short = reviews[word_count.between(MIN_WORDS, MAX_WORDS)]

# Balanced, so accuracy and the macro F1 mean the same thing and a coin flip scores 0.5.
sample = (
    short.groupby("label", group_keys=False)
    .sample(PER_CLASS, random_state=SEED)
    .sample(frac=1, random_state=SEED)  # shuffle, so the file is not sorted by label
)

# 0 and 1 are what the source uses; the words are what the notebook reads.
sample["sentiment"] = sample["label"].map({0: "negative", 1: "positive"})
sample[["sentiment", "text"]].to_csv("reviews.csv", index=False)

print(f"{len(sample)} reviews written to reviews.csv")
print(sample["sentiment"].value_counts().to_string())
print(f"median length: {sample['text'].str.split().str.len().median():.0f} words")
