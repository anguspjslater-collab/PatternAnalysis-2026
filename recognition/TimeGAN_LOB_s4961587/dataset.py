from pathlib import Path

def find_lobster_files(data_dir: Path, ticker="AMZN", level=10):
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

def load_raw():
    """
    Load the raw message and orderbook files into a single DataFrame
    """
    pass

def check_invariants():
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