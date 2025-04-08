import os
import pandas as pd
from tqdm import tqdm

# Ensure progress_apply works
tqdm.pandas(desc="Checking files exist")

def check_image_files(df, img_root):
    original_len = len(df)

    def file_exists(row):
        return os.path.exists(os.path.join(img_root, row['filename']))

    df['exists'] = df.progress_apply(file_exists, axis=1)
    df = df[df['exists'] == True].drop(columns='exists').reset_index(drop=True)

    new_len = len(df)
    deleted_rows = original_len - new_len

    print(f"Original rows: {original_len}")
    print(f"Remaining rows: {new_len}")
    print(f"Deleted rows (missing files): {deleted_rows}")

    return df
