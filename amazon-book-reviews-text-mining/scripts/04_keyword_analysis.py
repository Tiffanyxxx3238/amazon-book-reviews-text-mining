import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ==================================================
# 1. 路徑與輸出資料夾設定
# ==================================================
INPUT_PATH = "data/processed/preprocessed_reviews.csv"
OUTPUT_DIR = "outputs/keyword_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# 2. 分析參數設定
# ==================================================
TOP_N = 15
MIN_DF = 2
MAX_DF = 0.90

# 分組門檻
POSITIVE_THRESHOLD = 4   # score >= 4
NEGATIVE_THRESHOLD = 2   # score <= 2

# 你同學 preprocessing 已做過一般 stopwords removal，
# 這裡再加「書評領域太常見、但分析價值較低」的 domain stopwords
CUSTOM_DOMAIN_STOPWORDS = {
    "book", "read", "reading", "one"
}

# ngram_range=(1,1) 代表只看單字
# 如果之後想做詞組，可改成 (1,2)
NGRAM_RANGE = (1, 1)

# ==================================================
# 3. 讀取資料
# ==================================================
df = pd.read_csv(INPUT_PATH)

print("===== Data Loaded =====")
print("Columns:", df.columns.tolist())
print("Original rows:", len(df))
print()

required_cols = ["review_text", "review_score", "clean_text"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# 保險處理
df = df.dropna(subset=["clean_text", "review_score"])
df["clean_text"] = df["clean_text"].astype(str).str.strip()
df = df[df["clean_text"] != ""]
df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")
df = df.dropna(subset=["review_score"])

print("===== After Basic Filtering =====")
print("Remaining rows:", len(df))
print()

# ==================================================
# 4. 分組
# ==================================================
all_df = df.copy()
positive_df = df[df["review_score"] >= POSITIVE_THRESHOLD].copy()
negative_df = df[df["review_score"] <= NEGATIVE_THRESHOLD].copy()

groups = {
    "all_reviews": all_df,
    "positive_reviews": positive_df,
    "negative_reviews": negative_df
}

print("===== Group Sizes =====")
for group_name, group_df in groups.items():
    print(f"{group_name}: {len(group_df)}")
print()

# ==================================================
# 5. TF 函式
# ==================================================
def get_top_tf_keywords(texts, top_n=15, min_df=2, max_df=0.90,
                        custom_stopwords=None, ngram_range=(1, 1)):
    if len(texts) == 0:
        return pd.DataFrame(columns=["rank", "keyword", "score"])

    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        stop_words=list(custom_stopwords) if custom_stopwords else None,
        ngram_range=ngram_range
    )

    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    term_sums = X.sum(axis=0).A1

    result_df = pd.DataFrame({
        "keyword": terms,
        "score": term_sums
    })

    result_df = result_df.sort_values(by="score", ascending=False).head(top_n).reset_index(drop=True)
    result_df.insert(0, "rank", range(1, len(result_df) + 1))
    return result_df

# ==================================================
# 6. TF-IDF 函式
# ==================================================
def get_top_tfidf_keywords(texts, top_n=15, min_df=2, max_df=0.90,
                           custom_stopwords=None, ngram_range=(1, 1)):
    if len(texts) == 0:
        return pd.DataFrame(columns=["rank", "keyword", "score"])

    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        stop_words=list(custom_stopwords) if custom_stopwords else None,
        ngram_range=ngram_range
    )

    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    # 用平均 TF-IDF 當排序依據
    term_means = X.mean(axis=0).A1

    result_df = pd.DataFrame({
        "keyword": terms,
        "score": term_means
    })

    result_df = result_df.sort_values(by="score", ascending=False).head(top_n).reset_index(drop=True)
    result_df.insert(0, "rank", range(1, len(result_df) + 1))
    return result_df

# ==================================================
# 7. 跑 all / positive / negative 的 TF、TF-IDF
# ==================================================
all_tf_results = {}
all_tfidf_results = {}

for group_name, group_df in groups.items():
    texts = group_df["clean_text"].tolist()

    tf_df = get_top_tf_keywords(
        texts=texts,
        top_n=TOP_N,
        min_df=MIN_DF,
        max_df=MAX_DF,
        custom_stopwords=CUSTOM_DOMAIN_STOPWORDS,
        ngram_range=NGRAM_RANGE
    )

    tfidf_df = get_top_tfidf_keywords(
        texts=texts,
        top_n=TOP_N,
        min_df=MIN_DF,
        max_df=MAX_DF,
        custom_stopwords=CUSTOM_DOMAIN_STOPWORDS,
        ngram_range=NGRAM_RANGE
    )

    all_tf_results[group_name] = tf_df
    all_tfidf_results[group_name] = tfidf_df

    tf_path = os.path.join(OUTPUT_DIR, f"tf_results_{group_name}.csv")
    tfidf_path = os.path.join(OUTPUT_DIR, f"tfidf_results_{group_name}.csv")

    tf_df.to_csv(tf_path, index=False, encoding="utf-8-sig")
    tfidf_df.to_csv(tfidf_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {tf_path}")
    print(f"Saved: {tfidf_path}")
    print()

# ==================================================
# 8. TF vs TF-IDF 比較表
# ==================================================
def build_comparison_table(tf_df, tfidf_df):
    max_len = max(len(tf_df), len(tfidf_df))

    tf_keywords = tf_df["keyword"].tolist() + [""] * (max_len - len(tf_df))
    tf_scores = tf_df["score"].tolist() + [""] * (max_len - len(tf_df))

    tfidf_keywords = tfidf_df["keyword"].tolist() + [""] * (max_len - len(tfidf_df))
    tfidf_scores = tfidf_df["score"].tolist() + [""] * (max_len - len(tfidf_df))

    compare_df = pd.DataFrame({
        "rank": range(1, max_len + 1),
        "tf_keyword": tf_keywords,
        "tf_score": tf_scores,
        "tfidf_keyword": tfidf_keywords,
        "tfidf_score": tfidf_scores
    })
    return compare_df

for group_name in groups.keys():
    compare_df = build_comparison_table(
        all_tf_results[group_name],
        all_tfidf_results[group_name]
    )
    compare_path = os.path.join(OUTPUT_DIR, f"comparison_tf_vs_tfidf_{group_name}.csv")
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {compare_path}")

print()

# ==================================================
# 9. 合併成總表
# ==================================================
combined_rows = []

for group_name in groups.keys():
    tf_df = all_tf_results[group_name].copy()
    tf_df["group"] = group_name
    tf_df["method"] = "TF"

    tfidf_df = all_tfidf_results[group_name].copy()
    tfidf_df["group"] = group_name
    tfidf_df["method"] = "TF-IDF"

    combined_rows.append(tf_df)
    combined_rows.append(tfidf_df)

combined_df = pd.concat(combined_rows, ignore_index=True)
combined_df = combined_df[["group", "method", "rank", "keyword", "score"]]

combined_path = os.path.join(OUTPUT_DIR, "all_keyword_results_combined.csv")
combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")
print(f"Saved: {combined_path}")
print()

# ==================================================
# 10. 產生分析摘要
# ==================================================
def generate_analysis_summary(groups, all_tf_results, all_tfidf_results):
    lines = []
    lines.append("Keyword Analysis Summary")
    lines.append("=" * 70)
    lines.append("")

    lines.append("1. Corpus Overview")
    for group_name, group_df in groups.items():
        lines.append(f"- {group_name}: {len(group_df)} documents")
    lines.append("")

    lines.append("2. Top Keywords by Method")
    for group_name in groups.keys():
        lines.append(f"\n[{group_name}]")

        tf_keywords = all_tf_results[group_name]["keyword"].tolist()
        tfidf_keywords = all_tfidf_results[group_name]["keyword"].tolist()

        lines.append("TF top keywords:")
        lines.append(", ".join(tf_keywords) if tf_keywords else "No result")

        lines.append("TF-IDF top keywords:")
        lines.append(", ".join(tfidf_keywords) if tfidf_keywords else "No result")
    lines.append("")

    lines.append("3. Method Comparison")
    for group_name in groups.keys():
        tf_set = set(all_tf_results[group_name]["keyword"].tolist())
        tfidf_set = set(all_tfidf_results[group_name]["keyword"].tolist())

        overlap = sorted(tf_set & tfidf_set)
        tf_only = sorted(tf_set - tfidf_set)
        tfidf_only = sorted(tfidf_set - tf_set)

        lines.append(f"\n[{group_name}]")
        lines.append(f"- Overlap keywords: {', '.join(overlap) if overlap else 'None'}")
        lines.append(f"- TF-only keywords: {', '.join(tf_only) if tf_only else 'None'}")
        lines.append(f"- TF-IDF-only keywords: {', '.join(tfidf_only) if tfidf_only else 'None'}")

    lines.append("")
    lines.append("4. Initial Interpretation")
    lines.append("- TF mainly highlights high-frequency words in each review group.")
    lines.append("- TF-IDF highlights more distinctive words that better represent specific reviews or sentiment groups.")
    lines.append("- Positive and negative reviews may contain different keyword patterns, which helps reveal sentiment-related characteristics.")
    lines.append("- Removing domain-generic words such as 'book' or 'read' helps the analysis focus more on meaningful descriptive terms.")
    lines.append("")

    return "\n".join(lines)

summary_text = generate_analysis_summary(groups, all_tf_results, all_tfidf_results)

summary_path = os.path.join(OUTPUT_DIR, "analysis_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_text)

print(f"Saved: {summary_path}")

# ==================================================
# 11. 產生簡報版重點
# ==================================================
presentation_lines = [
    "Presentation-ready Findings",
    "=" * 70,
    "",
    "1. TF is frequency-based, so it often highlights common words that appear repeatedly in the corpus.",
    "2. TF-IDF is more effective for identifying words that are more distinctive and representative.",
    "3. Positive reviews tend to include more praise-related words.",
    "4. Negative reviews tend to include more complaint- or dissatisfaction-related words.",
    "5. Grouping reviews into all / positive / negative helps reveal clearer keyword patterns.",
    "6. Domain-specific stopwords were removed in the analysis stage to improve keyword quality.",
    ""
]

presentation_path = os.path.join(OUTPUT_DIR, "presentation_findings.txt")
with open(presentation_path, "w", encoding="utf-8") as f:
    f.write("\n".join(presentation_lines))

print(f"Saved: {presentation_path}")
print()
print("===== Done =====")
print("All keyword analysis outputs are saved in:", OUTPUT_DIR)