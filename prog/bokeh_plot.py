
import argparse
import sys
from pathlib import Path

from bokeh.layouts import row, column


import bokeh
from bokeh.server.server import Server
from bokeh.core import validation

import data_gen as dg
from source_data import RegressionSourceData, ClassificationSourceData
from general_layout import ClassifierGeneralLayout, RegressionGeneralLayout
from constants import CLUSTER_SIZE_DEF, CLUSTER_VOL_DEF, CLUSTERS_COUNT_DEF, PALETTE
from in_n_out import read_df


# TODO: @numba.njit()
# TODO: https://docs.bokeh.org/en/latest/docs/reference/core/templates.html?fbclid=IwAR1AzUzA2gmpmO3bGqtM7bpNrop-bzbHA3jjgP786VuJirgANO8m7Ia5qAk

# TODO: configurak v JSONu
# TODO: html stranky navic k proklikani a jako info

# TODO: sjednotit jmena (cluster size, density...)
# TODO: generictejsi generovani dat (volba alphy a bety u betavaribale)

validation.silence(1002, True)  # silence bokeh plot warning


def parse_args():

    parser = argparse.ArgumentParser(description="bokeh plot")
    parser.add_argument('--dataset', default='', nargs=1)
    parser.add_argument('--cols', default=['x', 'y', 'classification'], nargs='+', help='x y classification')
    parser.add_argument('--ver', choices=['cls', 'reg'], required=True,
                        help='cls for classification, reg for regression')

    parsed = parser.parse_args(sys.argv[1:])

    return parsed.dataset, parsed.cols, parsed.ver


if __name__ == '__main__':

    args = parse_args()
    column_names = args[1]
    if args[0] == '':
        df = dg.cluster_data_pandas(column_names=column_names,
                                    x_interval=(0, 30), y_interval=(-10, 10),
                                    clusters=CLUSTERS_COUNT_DEF, av_cluster_size=CLUSTER_SIZE_DEF,
                                    clust_size_vol=CLUSTER_VOL_DEF)  # TODO: hezceji
    else:
        path = args[0][0]
        df = read_df(path, column_names)

    version = args[2]

    def bkapp(doc):

        if version == 'reg':
            source_data = RegressionSourceData(df=df)
            lay = RegressionGeneralLayout(source_data=source_data)
        elif version == 'cls':
            source_data = ClassificationSourceData(df=df, palette=PALETTE)
            lay = ClassifierGeneralLayout(source_data=source_data)
        else:
            lay = row()

        doc.add_root(lay.layout)

    server = Server({'/': bkapp}, num_procs=1)
    server.start()

    server.io_loop.add_callback(server.show, "/")
    try:
        server.io_loop.start()
    except KeyboardInterrupt:
        print()
        print("Session closed.")
