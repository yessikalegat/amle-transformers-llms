# Data

| File | Rows | Columns |
|---|---|---|
| `poems.csv` | 1,082 | `author`, `title`, `text`, one poem or poem section per row |

Notebook 01 uses the first 200 rows to measure what a corpus costs in tokens. The file was
derived from the two Project Gutenberg texts below: straight quotes, no italic underscores
and no double hyphens, with each book split into its individual poems.

## Licences

- **Poems by Emily Dickinson**: *Poems by Emily Dickinson, Three Series, Complete* (1890 to 1896), [Project Gutenberg eBook #12242](https://www.gutenberg.org/ebooks/12242), public domain in the USA.
- **Leaves of Grass by Walt Whitman**: [Project Gutenberg eBook #1322](https://www.gutenberg.org/ebooks/1322), public domain in the USA.

Project Gutenberg texts are free in the USA and in most of the world. Readers elsewhere
should check their own country's copyright term before redistributing them.

## reviews.csv

| File | Rows | Columns |
|---|---|---|
| `reviews.csv` | 3,000 | `sentiment` (`positive` / `negative`, 1,500 each), `text` |
| `build_reviews.py` | n/a | The script that derives it from the source below |

Notebook 02 trains on 500 of these and scores on the other 2,500. The reviews are **short** on
purpose (28 words in the middle of the distribution) because on short texts a bag of words
finds little shared vocabulary and a pretrained encoder performs clearly better. Long reviews
repeat their words often enough for TF-IDF to perform as well, and the difference disappears.

Rebuild it with `uv run --with pyarrow python build_reviews.py`. The script downloads one
117 MB file from the hub; students never need to run it.

- **Amazon Polarity**: Zhang, X., Zhao, J. and LeCun, Y. (2015), [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626), from the Amazon review corpus of McAuley and Leskovec. Distributed as [`fancyzhx/amazon_polarity`](https://huggingface.co/datasets/fancyzhx/amazon_polarity), **Apache-2.0**.
