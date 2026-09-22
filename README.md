# Lexicon-based vs. Transformer-based Sentiment Analysis

Compares two lexicon-based methods (VADER, TextBlob) with a transformer (DistilBERT fine-tuned on SST-2) on positive/negative sentiment classification. Two test sets are used:

- SST-2 validation set: 872 movie review sentences. This is the domain DistilBERT was fine-tuned on.
- TweetEval sentiment test set, positive and negative tweets only: 6,347 tweets. DistilBERT has not seen this data. VADER was designed for social media text.

## Files

- `sentiment_comparison.ipynb`: the whole experiment (data, models, metrics, McNemar tests, error analysis), saved with its outputs
- `error_sample_labels.csv`: hand-assigned error types for the random error sample (notebook section 6.2)
- `results/`: everything the notebook writes (predictions, metrics, significance tests, error analysis tables, figures as PNG and PDF)
- `requirements.txt`: package versions used for the results

## How to run

Tested with Python 3.12.7 (Anaconda) on Windows with an NVIDIA GTX 1660 Ti. It also runs on CPU, just slower.

1. For GPU support, install PyTorch with CUDA from https://pytorch.org first (the results used torch 2.5.1 with CUDA 12.1).
2. `pip install -r requirements.txt`
3. Open `sentiment_comparison.ipynb` and run all cells. The datasets and the model are downloaded from the Hugging Face Hub on the first run.

A full run takes under two minutes on the GPU, not counting downloads. All steps are deterministic (the error sample uses `random_state=42`), so re-running gives the same numbers.

Datasets: [`stanfordnlp/sst2`](https://huggingface.co/datasets/stanfordnlp/sst2), [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (config `sentiment`)
Model: [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)

## Main results

| Dataset | Model | Accuracy | Macro-F1 |
|---|---|---|---|
| SST-2 | VADER | 0.631 | 0.607 |
| SST-2 | TextBlob | 0.628 | 0.605 |
| SST-2 | DistilBERT | 0.911 | 0.910 |
| Tweets | VADER | 0.704 | 0.704 |
| Tweets | TextBlob | 0.560 | 0.551 |
| Tweets | DistilBERT | 0.813 | 0.794 |

Full tables are in `results/metrics.csv` and `results/mcnemar.csv`.
