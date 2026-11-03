# Transformers & Large Language Models

A language model never sees a word. It sees a sequence of token ids, turns each one into a vector, and works on the vectors. In these notebooks you split one difficult sentence with four trained tokenizers and compare where they disagree, read the embedding table out of BERT, and examine the two kinds of model built on it: the one that classifies 2,500 Amazon product reviews from 500 labeled examples and then from none at all, and the one that writes.

## Learning Objectives

By the end of this repository, you should be able to:

- Compare how different tokenizers split a text.
- Measure static and contextual embeddings with cosine similarity.
- Plot attention weights in an encoder model.
- Classify texts with embeddings and zero-shot labels.
- Generate text from logits using temperature sampling.
- Compare the accuracy and cost of prompting and embeddings.

## Learning Path

Work through the notebooks in this order.

**Scope** marks the core path every student is expected to complete. Optional lessons stay in
the repository and are worth returning to, but the day does not depend on them. **Units** are
indicative pacing, where one unit is 45 minutes.

| File / Folder | Description | Scope | Units |
|---|---|---|---|
| [**01 - Tokens and Embeddings**](01_tokens_and_embeddings.ipynb) | Split one difficult sentence with four trained tokenizers, measure a corpus's cost in tokens, read BERT's embedding lookup table and compare one word's vectors in two sentences. | Core | 2.25 |
| [**02 - Encoder Models**](02_encoder_models.ipynb) | Recover a masked word, read the attention weights that put context into a vector, embed whole sentences, and beat TF-IDF on 500 labels, then on none at all. | Core | 2.5 |
| [**03 - Decoder Models**](03_decoder_models.ipynb) | Count a 3.8-billion-parameter model from its config, turn logits into words, reshape the distribution with temperature, and classify reviews by asking in English. It takes a long time to run on a laptop CPU, in the order of 10 minutes. | Core | 2.75 |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**Solutions**](solutions/) | Worked solutions for every notebook. |
| [**Data**](data/) | The poems the tokenizers are priced on, the product reviews every classifier is scored on, and their sources. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a
> **placeholder**. Replace it, including the `< >` brackets, with your own
> value. For example, `cd <repo-name>` becomes `cd amle-transformers-llms`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies and creates a virtual environment in `.venv/`. PyTorch is a large package and the environment comes to about 1 GB, so the first install takes a few minutes.

```bash
cd <repo-name>
uv sync
```

> [!NOTE]
> The lock file pins the **CPU-only** build of PyTorch, from PyTorch's own package
> index rather than PyPI. The default build bundles the NVIDIA CUDA runtime, several
> gigabytes of it, which is of no use on a machine without an NVIDIA GPU. Nothing in
> this repository needs a GPU.

---

### 5. Open the Notebooks

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open a notebook and select the Python environment created by `uv sync` as the kernel.

## Data and Credits

The notebooks download their models from the [Hugging Face model hub](https://huggingface.co/) on first run and cache them in `~/.cache/huggingface`, outside this repository. No account and no token are needed, and the hub prints one note about unauthenticated requests the first time, which is expected.

| Notebook | Downloads | What it fetches |
|---|---|---|
| 01 | **428 MB** | `bert-base-uncased`, plus three tokenizers at a few megabytes |
| 02 | **321 MB** | `gte-small` 65 MB, `distilbert-...-sst-2-english` 256 MB |
| 03 | **954 MB** | `Qwen2.5-0.5B-Instruct`; Phi-3's configuration is about a kilobyte |
| **Total** | **~1.7 GB** | plus about 360 MB of Python packages at `uv sync` |

The data is in `data/` and [data/README.md](data/README.md) lists every file with its license.

- **`bert-base-uncased`** and **`bert-base-cased`**: Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019), [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), Apache-2.0.
- **`gpt2`**: Radford, A. et al. (2019), [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), MIT license.
- **`google/flan-t5-small`**: Chung, H. W. et al. (2022), [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416), Apache-2.0.
- **`thenlper/gte-small`**: Li, Z. et al. (2023), [Towards General Text Embeddings with Multi-stage Contrastive Learning](https://arxiv.org/abs/2308.03281), MIT license.
- **`distilbert-base-uncased-finetuned-sst-2-english`**: DistilBERT (Sanh, V. et al., 2019, [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)) fine-tuned on SST-2, Apache-2.0.
- **`Qwen/Qwen2.5-0.5B-Instruct`**: Qwen Team (2024), [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115), Apache-2.0. Notebook 03 generates with it because the book's `Phi-3-mini-4k-instruct` needs a GPU; Phi-3's configuration file is still read, which costs a kilobyte.
- **Amazon Polarity**: Zhang, X., Zhao, J. and LeCun, Y. (2015), [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626), built from the Amazon review corpus of McAuley and Leskovec; distributed on the Hugging Face hub as [`fancyzhx/amazon_polarity`](https://huggingface.co/datasets/fancyzhx/amazon_polarity) under Apache-2.0. `data/reviews.csv` is a 3,000-row balanced sample of short reviews, built by `data/build_reviews.py`.
- **Poems by Emily Dickinson** and **Leaves of Grass** by Walt Whitman: [Project Gutenberg](https://www.gutenberg.org/) eBooks [#12242](https://www.gutenberg.org/ebooks/12242) and [#1322](https://www.gutenberg.org/ebooks/1322), public domain in the USA.
- The notebooks follow chapters 1 to 4 and 11 of **Jay Alammar and Maarten Grootendorst, *Hands-On Large Language Models*** (O'Reilly, 2024): its comparison of trained tokenizers, its contextual embeddings, its split between representation and generative models, its embedding-based and zero-shot classifiers, and its walk through a decoder's forward pass. The book's [companion code](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) is Apache-2.0.

## References & Further Reading

- [**The Illustrated Transformer**](https://jalammar.github.io/illustrated-transformer/): Jay Alammar's visual walk through the architecture, and the companion piece to the book.
- [**Transformers, the tech behind LLMs**](https://www.youtube.com/watch?v=wjZofJX0v4M): 3Blue1Brown on what a transformer does to a sequence of embeddings, in animation.
- [**Attention in transformers, step-by-step**](https://www.youtube.com/watch?v=eMlx5fFNoYc): the same series on the mechanism itself, derived rather than asserted.
- [**Hugging Face: Tokenizers**](https://huggingface.co/docs/transformers/en/tokenizer_summary): what WordPiece, BPE, SentencePiece and Unigram each do differently.
- [**The Tokenizer Playground**](https://huggingface.co/spaces/Xenova/the-tokenizer-playground): paste any text and see how several models cut it, in the browser.
- [**Hugging Face: Auto classes**](https://huggingface.co/docs/transformers/en/model_doc/auto): `AutoTokenizer`, `AutoModel` and the rest of the loaders the notebooks use.

### The papers behind the notebooks

- Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762): the transformer, and the attention that turns a static vector into a contextual one.
- Sennrich, R., Haddow, B. and Birch, A. (2016). [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909): byte pair encoding, the method behind GPT-2's vocabulary.
- Devlin, J. et al. (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805): the model notebooks 01 and 02 take apart, and the masked language modeling that trained it.
- Clark, K. et al. (2019). [What Does BERT Look At? An Analysis of BERT's Attention](https://arxiv.org/abs/1906.04341): why so many heads end up attending to `[SEP]`, which notebook 02 reproduces.
- Reimers, N. and Gurevych, I. (2019). [Sentence-BERT](https://arxiv.org/abs/1908.10084): why a sentence embedding needs a model trained for it rather than an average of BERT's token vectors.
- Brown, T. et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165): the paper that made few-shot prompting a method. It was written for base models, which is why notebook 03's completion-style block fails on a chat model.
- Holtzman, A. et al. (2020). [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751): what temperature, top-k and nucleus sampling each do to generated text.
