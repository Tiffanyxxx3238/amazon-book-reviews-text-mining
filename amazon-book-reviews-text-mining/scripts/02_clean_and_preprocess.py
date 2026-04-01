import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

input_path = "data/raw/Books_rating.csv"
output_path = "data/processed/preprocessed_reviews.csv"

# Load only needed columns and only part of the file
df = pd.read_csv(
    input_path,
    usecols=["review/text", "review/score"],
    nrows=10000
)

# Rename columns
df = df.rename(columns={
    "review/text": "review_text",
    "review/score": "review_score"
})

# Keep only needed non-null rows
df = df.dropna(subset=["review_text", "review_score"])

# Convert to string
df["review_text"] = df["review_text"].astype(str)

# Remove duplicates
df = df.drop_duplicates(subset=["review_text"])

# Remove very short reviews
df = df[df["review_text"].apply(lambda x: len(x.split()) >= 5)]

# Sample 500 reviews
if len(df) > 500:
    df = df.sample(n=500, random_state=42)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Apply preprocessing
df["clean_text"] = df["review_text"].apply(preprocess_text)

# Remove empty processed text
df = df[df["clean_text"].str.strip() != ""]

# Save file
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("Preprocessing completed.")
print("Remaining documents:", len(df))
print("Saved to:", output_path)

print("\nSample results:")
print(df[["review_text", "clean_text"]].head(5))