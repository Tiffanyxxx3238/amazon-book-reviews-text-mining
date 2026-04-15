# Amazon Book Reviews — Text Mining & Keyword Analysis

A text mining pipeline applied to Amazon book reviews. The project preprocesses raw review text, extracts keywords using TF and TF-IDF, and compares keyword patterns across all reviews, positive reviews (score ≥ 4), and negative reviews (score ≤ 2).

**Dataset:** [Amazon Books Rating](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews) (Kaggle) — `Books_rating.csv`  
**Language:** Python 3  
**Key libraries:** pandas · scikit-learn · NLTK · matplotlib

---

## Pipeline Overview

```
data/raw/Books_rating.csv
        │
        ▼
01_inspect_data.py          → preview columns, shape, sample rows
        │
        ▼
02_clean_and_preprocess.py  → filter, deduplicate, lemmatize → preprocessed_reviews.csv
        │
        ▼
03_export_examples.py       → save 5 before/after preprocessing examples → .txt
        │
        ▼
04_keyword_analysis.py      → TF & TF-IDF per group → CSVs + summary
        │
        ▼
05_plot_keywords.py         → horizontal bar charts per group/method → PNGs
```

---

## Project Structure

```
amazon-book-reviews-text-mining/
│
├── data/
│   ├── raw/
│   │   └── Books_rating.csv          # original dataset (not committed — see below)
│   └── processed/
│       └── preprocessed_reviews.csv  # output of step 02
│
├── outputs/
│   ├── preprocessing_examples.txt    # output of step 03
│   └── keyword_analysis/
│       ├── tf_results_all_reviews.csv
│       ├── tf_results_positive_reviews.csv
│       ├── tf_results_negative_reviews.csv
│       ├── tfidf_results_*.csv
│       ├── comparison_tf_vs_tfidf_*.csv
│       ├── all_keyword_results_combined.csv
│       ├── analysis_summary.txt
│       ├── presentation_findings.txt
│       └── plots/
│           ├── tf_all_reviews.png
│           ├── tfidf_all_reviews.png
│           ├── tf_positive_reviews.png
│           └── ...
│
├── 01_inspect_data.py
├── 02_clean_and_preprocess.py
├── 03_export_examples.py
├── 04_keyword_analysis.py
├── 05_plot_keywords.py
├── download_nltk_data.py
└── README.md
```

---

## Setup & Usage

**1. Install dependencies**
```bash
pip install pandas scikit-learn nltk matplotlib
```

**2. Download NLTK data** (first time only)
```bash
python download_nltk_data.py
```

**3. Get the dataset**

Download `Books_rating.csv` from [Kaggle](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews) and place it at `data/raw/Books_rating.csv`.

> The raw dataset is not committed to this repo due to file size. The processed output (`preprocessed_reviews.csv`) is included.

**4. Run the pipeline in order**
```bash
python 01_inspect_data.py
python 02_clean_and_preprocess.py
python 03_export_examples.py
python 04_keyword_analysis.py
python 05_plot_keywords.py
```

---

## Preprocessing Steps (step 02)

Applied to the first 10,000 rows; 500 reviews sampled for analysis.

| Step | Description |
|------|-------------|
| Column selection | Keep `review/text` and `review/score` only |
| Null removal | Drop rows missing text or score |
| Deduplication | Remove exact duplicate review texts |
| Length filter | Remove reviews with fewer than 5 words |
| Lowercasing | Normalize to lowercase |
| Digit removal | Strip all numeric characters |
| Punctuation removal | Strip all punctuation |
| Stopword removal | NLTK English stopwords |
| Lemmatization | WordNet lemmatizer |

---

## Keyword Analysis (step 04)

Reviews are split into three groups:

| Group | Condition |
|-------|-----------|
| All reviews | all 500 sampled reviews |
| Positive reviews | score ≥ 4 |
| Negative reviews | score ≤ 2 |

Two methods are applied and compared for each group:

**TF (Term Frequency)** — ranks keywords by raw count across all documents in the group. Highlights the most frequently mentioned words overall.

**TF-IDF (Term Frequency–Inverse Document Frequency)** — ranks keywords by average TF-IDF score. Downweights words that appear in nearly every review, surfacing more distinctive and group-specific terms.

Domain stopwords (`book`, `read`, `reading`, `one`) are additionally removed at this stage since they appear across all groups and carry low discriminative value.

**Parameters:**
- `top_n = 15` keywords per method per group
- `min_df = 2` (word must appear in at least 2 documents)
- `max_df = 0.90` (word must not appear in more than 90% of documents)
- `ngram_range = (1, 1)` — unigrams only

---

## Output Files

| File | Description |
|------|-------------|
| `tf_results_{group}.csv` | Top-15 TF keywords for each group |
| `tfidf_results_{group}.csv` | Top-15 TF-IDF keywords for each group |
| `comparison_tf_vs_tfidf_{group}.csv` | Side-by-side TF vs TF-IDF ranking |
| `all_keyword_results_combined.csv` | All groups and methods in one table |
| `analysis_summary.txt` | Overlap/difference analysis between methods |
| `presentation_findings.txt` | Concise bullet-point findings |
| `plots/*.png` | Horizontal bar charts (300 dpi) |

---

## Key Findings

- **TF** tends to surface high-frequency general words (e.g., *story*, *great*, *like*) that are common across all review types.
- **TF-IDF** surfaces more distinctive terms — words that matter more to a specific group (positive vs. negative) rather than the corpus as a whole.
- **Positive reviews** emphasize language around enjoyment, recommendation, and emotional engagement.
- **Negative reviews** emphasize language around disappointment, unmet expectations, and specific complaints.
- Removing domain stopwords (`book`, `read`) meaningfully improves keyword quality by preventing trivially common book-review vocabulary from dominating the rankings.

---

## Concepts Demonstrated

- Text preprocessing pipeline: normalization, stopword removal, lemmatization
- Bag-of-words representation with `CountVectorizer`
- TF-IDF weighting with `TfidfVectorizer`
- Sentiment-based corpus segmentation
- Comparison of frequency-based vs. importance-weighted keyword extraction
- Automated output generation: CSVs, text summaries, bar chart visualizations
