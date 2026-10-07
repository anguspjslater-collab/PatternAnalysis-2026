from pathlib import Path
import pandas as pd

def find_lobster_files(data_dir: Path, ticker: str = "AMZN", level: int = 10):
    """
    Locate the message and orderbook files for a given stock
    """
    data_dir = Path(data_dir).expanduser()

    pattern = f"{ticker}_2012-06-21_34200000_57600000_message_{level}.csv"
    matches = sorted(data_dir.rglob(pattern))

    if len(matches) != 1: 
        raise FileNotFoundError(
            f"Expected 1 file matching {pattern} in {data_dir}, found {len(matches)}: {matches}")

    msg_path = matches[0] # location of message file
    ob_path = msg_path.with_name(msg_path.name.replace("message", "orderbook")) # location of orderbook file

    if not ob_path.exists():
        raise FileNotFoundError(
            f"Expected orderbook file {ob_path} to exist, but it does not")

    return msg_path, ob_path

def load_raw(msg_path: Path, ob_path: Path, level: int = 10):
    """
    Load the raw message and orderbook files into a single DataFrame
    """
    msg_cols = ["time", "type", "order_id", "size", "price", "direction"]
    
    ob_cols = [] # Bid/ask 1 to 10 levels, each updated on an event (from message file)
    for lev in range(1, level + 1):
        ob_cols += [f"ask_p{lev}", f"ask_s{lev}", f"bid_p{lev}", f"bid_s{lev}"]

    msg = pd.read_csv(msg_path, header=None, names=msg_cols)
    ob = pd.read_csv(ob_path, header=None, names=ob_cols)

    if len(msg) != len(ob): # Ensure message and orderbook have the same number of rows
       raise ValueError(f"Row mismatch: {len(msg)} messages vs {len(ob)} orderbook rows")

    lob = pd.concat([msg, ob], axis=1) # Join message and orderbook side by side

    return lob

def check_invariants(lob: pd.DataFrame):
    """
    Check that the data obeys the invariants we want TimeGAN to learn
    """
    
    pass

def build_features():
    """
    Build the features for the TimeGAN model. 
    This includes: trim, resample, mid-relative representation, etc.
    """
    pass

def make_splits():
    """
    Split the data into train, validation, and test sets in chronological order
    Train: first 67% of the data - rounds to whole hours when first and last 15 minutes are trimmed.
    Validate: next 17% of 
    Test: last 17% of the data
    4-1-1 hours split
    """
    pass