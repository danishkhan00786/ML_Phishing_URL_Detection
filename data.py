"""Loads and cleans the dataset. Works with Kaggle-style files (URL, Label columns)."""
import pandas as pd

SAFE = {"good", "benign", "legitimate", "safe", "0"}

def load_data(path="data/urls.csv", limit=100000):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    url_col = next((c for c in ["url", "urls", "link"] if c in df.columns), None)
    label_col = next((c for c in ["label", "status", "class", "type", "result", "target"] if c in df.columns), None)
    if url_col is None or label_col is None:
        raise SystemExit(f"Could not find the URL and label columns. Your columns are: {list(df.columns)}")
    df = df[[url_col, label_col]].rename(columns={url_col: "url", label_col: "label"})
    df = df.dropna().drop_duplicates(subset="url")
    df = df.sample(min(len(df), limit), random_state=42)
    print("Label counts:\n", df["label"].value_counts().to_string())
    df = df.reset_index(drop=True)
    y = (~df["label"].astype(str).str.lower().isin(SAFE)).astype(int)   # 1 = phishing
    return df, y
