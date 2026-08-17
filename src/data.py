from pathlib import Path
import pandas as pd

DATA_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
DEFAULT_PATH = Path("data/BBBP.csv")


def load_bbbp(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Download BBBP if necessary and return a validated raw dataframe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        print(f"Downloading BBBP dataset to {path} ...")
        df = pd.read_csv(DATA_URL)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)

    required = {"smiles", "p_np"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df
