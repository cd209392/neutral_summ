from data.preprocess import TextProcessing

import pandas as pd
import time
import argparse
import re

DATA_DIR = "./data/"


def preprocess(file_path):
    processing = TextProcessing()
    df = pd.read_csv(file_path)
    st = time.perf_counter()
    
    print(f"Processing data at '{file_path}'")
    
    df["review"] = df["review"].apply(processing.preprocess)
    
    file_name = str(re.split(r"[\\/]", file_path)[-1])
    
    df.to_csv(f"{DATA_DIR}processed/{file_name}", index=False)
    
    print(f"Completed after {(time.perf_counter() - st):.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to raw data")
    # Parse arguments
    args = parser.parse_args()
    preprocess(file_path=args.source)