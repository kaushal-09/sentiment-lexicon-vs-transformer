# Lexicon-based vs. Transformer-based Sentiment Analysis

Compares VADER and TextBlob (lexicon-based) with DistilBERT fine-tuned on SST-2 on two test sets: the SST-2 validation set (872 movie review sentences, the domain DistilBERT was trained on) and the positive/negative tweets from the TweetEval sentiment test set (6,347 tweets, new to DistilBERT).

Everything is in `sentiment_comparison.ipynb`, saved with its outputs. `error_sample_labels.csv` has the hand-assigned error types for the error analysis and `results/` has everything the notebook writes (predictions, metrics, McNemar tests, figures).

## Running it

Used Python 3.12 (Anaconda) on Windows with a GTX 1660 Ti, but it also runs on CPU. For the GPU, install PyTorch with CUDA from pytorch.org first (torch 2.5.1 with CUDA 12.1 was used), then

    pip install -r requirements.txt

and run all cells. The datasets and the model come from the Hugging Face Hub: `stanfordnlp/sst2`, `cardiffnlp/tweet_eval` (sentiment) and `distilbert/distilbert-base-uncased-finetuned-sst-2-english`. Re-running the notebook gave the same numbers every time.

## Results

|                 | VADER | TextBlob | DistilBERT |
|-----------------|-------|----------|------------|
| SST-2 accuracy  | 0.631 | 0.628    | 0.911      |
| SST-2 macro-F1  | 0.607 | 0.605    | 0.910      |
| Tweets accuracy | 0.704 | 0.560    | 0.813      |
| Tweets macro-F1 | 0.704 | 0.551    | 0.794      |
