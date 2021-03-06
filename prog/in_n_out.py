
import os
from pathlib import Path
import pandas as pd

from constants import SAVED_DATASETS_DIR_NAME


# TODO: ukladat classy ve stringu


def save_source(col_data_source, name):
    """Save given ColumnDataSource as csv
    throws away last column because colors are expected to be there
    returns absolute path where the file was saved.
    """
    if name[-4:] != ".csv":
        name += ".csv"

    pandas_df = col_data_source.to_df()
    if 'color' in pandas_df.columns:
        pandas_df = pandas_df.drop(columns='color')

    dir_name = SAVED_DATASETS_DIR_NAME
    if dir_name[-1] != '/':
        dir_name += '/'
    dir_name += name[:name.rfind('/') + 1]
    name = name[name.rfind('/') + 1:]
    print(dir_name)
    print(name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    path = Path(dir_name)
    path = path / name

    pandas_df.to_csv(path, encoding='utf-8')

    return os.path.abspath(path)


def read_df(path, col_names):
    df = pd.read_csv(path)[col_names]

    if len(col_names) == 3:  # converting classes column to string
        cls_col_name = col_names[2]
        df[cls_col_name] = df[cls_col_name].astype(str)
    return df
