import pandas as pd

input_path = "data/processed/preprocessed_reviews.csv"
output_path = "outputs/preprocessing_examples.txt"

df = pd.read_csv(input_path)

examples = df[["review_text", "clean_text"]].head(5)

with open(output_path, "w", encoding="utf-8") as f:
    for i, row in examples.iterrows():
        f.write(f"Example {i+1}\n")
        f.write("Original Text:\n")
        f.write(str(row["review_text"]) + "\n")
        f.write("Processed Text:\n")
        f.write(str(row["clean_text"]) + "\n")
        f.write("-" * 50 + "\n")

print(examples)
print("Examples saved to:", output_path)