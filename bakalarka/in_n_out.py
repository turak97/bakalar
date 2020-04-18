
import os
from pathlib import Path
import pandas as pd

from constants import SAVED_DATASETS_DIR_NAME, SAVED_DATASET_FILE_NAME

# TODO: elegantni cislovani souboru
# TODO: ukladat classy ve stringu

def save_source(col_data_source):
    """
    save given ColumnDataSource as csv
    throws away last column because colors are expected to be there
    returns absolute path where the file was saved
    """
    pandas_df = col_data_source.to_df()
    pandas_df = pandas_df.drop(columns='color')

    if not os.path.exists(SAVED_DATASETS_DIR_NAME):
        os.makedirs(SAVED_DATASETS_DIR_NAME)
    path = Path(SAVED_DATASETS_DIR_NAME)
    path = path / SAVED_DATASET_FILE_NAME

    pandas_df.to_csv(path, encoding='utf-8')

    return os.path.abspath(path)


def read_df(path, col_names):
    df = pd.read_csv(path)[col_names]

    if len(col_names) == 3:  # converting classes column to string
        cls_col_name = col_names[2]
        df[cls_col_name] = df[cls_col_name].astype(str)
    return df