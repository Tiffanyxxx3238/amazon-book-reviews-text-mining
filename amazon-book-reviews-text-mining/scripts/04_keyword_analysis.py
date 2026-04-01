import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# =========================
# 1. 路徑設定
# =========================
INPUT_PATH = "data/processed/preprocessed_reviews.csv"
OUTPUT_DIR = "outputs/keyword_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2. 參數設定
# =========================
TOP_N = 15  # 每組輸出前幾個關鍵詞
MIN_DF = 2  # 至少在幾篇文件中出現過，避免太雜
MAX_DF = 0.9  # 太常見的詞排除
POSITIVE_THRESHOLD = 4  # review_score >= 4 視為 positive
NEGATIVE_THRESHOLD = 2  # review_score <= 2 視為 negative

# =========================
# 3. 讀取資料
# =========================
df = pd.read_csv(INPUT_PATH)

print("===== Data Loaded =====")
print("Columns:", df.columns.tolist())
print("Number of rows:", len(df))
print()

# 檢查必要欄位
required_cols = ["review_text", "review_score", "clean_text"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# 移除空值與空字串
df = df.dropna(subset=["clean_text", "review_score"])
df["clean_text"] = df["clean_text"].astype(str).str.strip()
df = df[df["clean_text"] != ""]

# 分數轉成數值
df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")
df = df.dropna(subset=["review_score"])

print("===== Cleaned for Analysis =====")
print("Remaining rows:", len(df))
print()

# =========================
# 4. 分組資料
# =========================
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

# =========================
# 5. 函式：取得 TF Top Keywords
# =========================
def get_top_tf_keywords(texts, top_n=15, min_df=2, max_df=0.9):
    """
    用 CountVectorizer 計算 TF（詞頻）
    回傳 DataFrame: rank, keyword, score
    """
    if len(texts) == 0:
        return pd.DataFrame(columns=["rank", "keyword", "score"])

    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(texts)

    # 所有文件的詞頻總和
    term_sums = X.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    result_df = pd.DataFrame({
        "keyword": terms,
        "score": term_sums
    })

    result_df = result_df.sort_values(by="score", ascending=False).head(top_n).reset_index(drop=True)
    result_df.insert(0, "rank", range(1, len(result_df) + 1))
    return result_df

# =========================
# 6. 函式：取得 TF-IDF Top Keywords
# =========================
def get_top_tfidf_keywords(texts, top_n=15, min_df=2, max_df=0.9):
    """
    用 TfidfVectorizer 計算 TF-IDF
    這裡用「每個詞在所有文件的平均 TF-IDF」做排序
    回傳 DataFrame: rank, keyword, score
    """
    if len(texts) == 0:
        return pd.DataFrame(columns=["rank", "keyword", "score"])

    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(texts)

    # 平均 TF-IDF
    term_means = X.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    result_df = pd.DataFrame({
        "keyword": terms,
        "score": term_means
    })

    result_df = result_df.sort_values(by="score", ascending=False).head(top_n).reset_index(drop=True)
    result_df.insert(0, "rank", range(1, len(result_df) + 1))
    return result_df

# =========================
# 7. 逐組分析並輸出 CSV
# =========================
all_tf_results = {}
all_tfidf_results = {}

for group_name, group_df in groups.items():
    texts = group_df["clean_text"].tolist()

    tf_df = get_top_tf_keywords(
        texts=texts,
        top_n=TOP_N,
        min_df=MIN_DF,
        max_df=MAX_DF
    )
    tfidf_df = get_top_tfidf_keywords(
        texts=texts,
        top_n=TOP_N,
        min_df=MIN_DF,
        max_df=MAX_DF
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

# =========================
# 8. 做方法比較表
# =========================
def compare_tf_and_tfidf(tf_df, tfidf_df, group_name):
    """
    將 TF 與 TF-IDF top keywords 並排比較
    """
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

    compare_path = os.path.join(OUTPUT_DIR, f"comparison_tf_vs_tfidf_{group_name}.csv")
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {compare_path}")

for group_name in groups.keys():
    compare_tf_and_tfidf(
        all_tf_results[group_name],
        all_tfidf_results[group_name],
        group_name
    )

print()

# =========================
# 9. 產生初步分析文字
# =========================
def generate_analysis_text(groups, all_tf_results, all_tfidf_results):
    """
    產出可以直接交給學姊整合的分析摘要文字
    """
    lines = []
    lines.append("Keyword Analysis Summary")
    lines.append("=" * 60)
    lines.append("")

    # 基本資料量
    lines.append("1. Corpus Overview")
    for group_name, group_df in groups.items():
        lines.append(f"- {group_name}: {len(group_df)} documents")
    lines.append("")

    # 各組 top 關鍵詞
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

    # 方法差異
    lines.append("3. Method Comparison")
    for group_name in groups.keys():
        tf_set = set(all_tf_results[group_name]["keyword"].tolist())
        tfidf_set = set(all_tfidf_results[group_name]["keyword"].tolist())

        overlap = tf_set.intersection(tfidf_set)
        tf_only = tf_set - tfidf_set
        tfidf_only = tfidf_set - tf_set

        lines.append(f"\n[{group_name}]")
        lines.append(f"- Overlap keywords: {', '.join(sorted(overlap)) if overlap else 'None'}")
        lines.append(f"- TF-only keywords: {', '.join(sorted(tf_only)) if tf_only else 'None'}")
        lines.append(f"- TF-IDF-only keywords: {', '.join(sorted(tfidf_only)) if tfidf_only else 'None'}")

    lines.append("")

    # 初步結論模板
    lines.append("4. Initial Findings")
    lines.append("- TF tends to highlight words that appear frequently across the corpus.")
    lines.append("- TF-IDF tends to highlight words that are more distinctive or representative in specific reviews.")
    lines.append("- If positive and negative groups show different keywords, this suggests that keyword extraction can capture sentiment-related patterns.")
    lines.append("- The comparison between TF and TF-IDF helps reveal the difference between frequency-based importance and distinctiveness-based importance.")
    lines.append("")

    return "\n".join(lines)

summary_text = generate_analysis_text(groups, all_tf_results, all_tfidf_results)

summary_path = os.path.join(OUTPUT_DIR, "analysis_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_text)

print(f"Saved: {summary_path}")
print()

# =========================
# 10. 額外輸出：整合版總表
# =========================
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

print("===== Done =====")
print("Your outputs are in:", OUTPUT_DIR)