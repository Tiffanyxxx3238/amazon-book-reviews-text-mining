import os
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# 1. 路徑設定
# ==================================================
INPUT_DIR = "outputs/keyword_analysis"
OUTPUT_DIR = "outputs/keyword_analysis/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# 2. 要處理的群組
# ==================================================
GROUPS = ["all_reviews", "positive_reviews", "negative_reviews"]

# ==================================================
# 3. 共用畫圖函式
# ==================================================
def plot_bar_chart(df, title, output_path, top_n=10):
    if df.empty:
        print(f"Skipped empty dataframe for: {title}")
        return

    df = df.head(top_n).copy()

    plt.figure(figsize=(10, 6))
    plt.barh(df["keyword"][::-1], df["score"][::-1])
    plt.xlabel("Score")
    plt.ylabel("Keyword")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot: {output_path}")

# ==================================================
# 4. 畫 TF 與 TF-IDF 各自長條圖
# ==================================================
for group_name in GROUPS:
    tf_path = os.path.join(INPUT_DIR, f"tf_results_{group_name}.csv")
    tfidf_path = os.path.join(INPUT_DIR, f"tfidf_results_{group_name}.csv")

    if os.path.exists(tf_path):
        tf_df = pd.read_csv(tf_path)
        plot_bar_chart(
            tf_df,
            title=f"TF Top Keywords - {group_name.replace('_', ' ').title()}",
            output_path=os.path.join(OUTPUT_DIR, f"tf_{group_name}.png"),
            top_n=10
        )

    if os.path.exists(tfidf_path):
        tfidf_df = pd.read_csv(tfidf_path)
        plot_bar_chart(
            tfidf_df,
            title=f"TF-IDF Top Keywords - {group_name.replace('_', ' ').title()}",
            output_path=os.path.join(OUTPUT_DIR, f"tfidf_{group_name}.png"),
            top_n=10
        )

# ==================================================
# 5. 畫 TF vs TF-IDF comparison 圖
#    做法：把兩種方法的 top 10 關鍵詞合併後畫成兩張圖
# ==================================================
def plot_comparison_keywords(group_name):
    tf_path = os.path.join(INPUT_DIR, f"tf_results_{group_name}.csv")
    tfidf_path = os.path.join(INPUT_DIR, f"tfidf_results_{group_name}.csv")

    if not os.path.exists(tf_path) or not os.path.exists(tfidf_path):
        print(f"Comparison skipped for {group_name}")
        return

    tf_df = pd.read_csv(tf_path).head(10).copy()
    tfidf_df = pd.read_csv(tfidf_path).head(10).copy()

    # TF
    plt.figure(figsize=(10, 6))
    plt.barh(tf_df["keyword"][::-1], tf_df["score"][::-1])
    plt.xlabel("TF Score")
    plt.ylabel("Keyword")
    plt.title(f"TF Keywords - {group_name.replace('_', ' ').title()}")
    plt.tight_layout()
    tf_output = os.path.join(OUTPUT_DIR, f"comparison_tf_{group_name}.png")
    plt.savefig(tf_output, dpi=300)
    plt.close()
    print(f"Saved plot: {tf_output}")

    # TF-IDF
    plt.figure(figsize=(10, 6))
    plt.barh(tfidf_df["keyword"][::-1], tfidf_df["score"][::-1])
    plt.xlabel("TF-IDF Score")
    plt.ylabel("Keyword")
    plt.title(f"TF-IDF Keywords - {group_name.replace('_', ' ').title()}")
    plt.tight_layout()
    tfidf_output = os.path.join(OUTPUT_DIR, f"comparison_tfidf_{group_name}.png")
    plt.savefig(tfidf_output, dpi=300)
    plt.close()
    print(f"Saved plot: {tfidf_output}")

for group_name in GROUPS:
    plot_comparison_keywords(group_name)

print("===== Plotting Finished =====")
print("All plots are saved in:", OUTPUT_DIR)