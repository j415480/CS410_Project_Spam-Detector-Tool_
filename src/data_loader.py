import pandas as pd
from pathlib import Path


def load_sms_spam_dataset(csv_path):
    """
    Load the SMS Spam Collection dataset from CSV or Kaggle and return a cleaned DataFrame.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, encoding="latin-1") #need to keep this because emojis and stuff in texts

    # other Kaggle spam datasets had junk "unnamed" column, this one didn't but user may DL other datasets
    extra_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if extra_cols:
        df = df.drop(columns=extra_cols)

    # column names = label and text (label 0/1, text is the name)
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})

    # Place into df to secure the dataset, no more moidifying it
    df = df[["label", "text"]].copy()

    # Drop rows with missing text to prevent abends
    df = df.dropna(subset=["text"])

    # Dataframe to store 0/1, where spam=1, ham=0
    df["target"] = (df["label"] == "spam").astype(int)

    return df
