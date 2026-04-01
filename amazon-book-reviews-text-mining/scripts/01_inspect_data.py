import pandas as pd

input_path = "data/raw/Books_rating.csv"

df = pd.read_csv(input_path)

print("Columns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nNumber of rows:", len(df))