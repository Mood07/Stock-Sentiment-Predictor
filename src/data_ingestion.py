import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads raw sentiment dataset from CSV.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=["sentiment", "sentence"],
        encoding="latin-1"
    )
    print(f"✅ Loaded dataset with shape: {df.shape}")
    return df
