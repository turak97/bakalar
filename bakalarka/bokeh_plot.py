
import argparse
import sys
from pathlib import Path

from bokeh.layouts import row, column


import bokeh
from bokeh.server.server import Server
from bokeh.core import validation

import data_gen as dg
from source_data import SourceData
from general_layout import ClassifierGeneralLayout, BasicGeneralLayout
from constants import POL_FROM_DGR, POL_TO_DGR, X_EXT, Y_EXT
from constants import CLUSTER_SIZE_DEF, CLUSTER_VOL_DEF, CLUSTERS_COUNT_DEF
from in_n_out import read_df

# TODO: Datasourcy podle trid, nova trida pro data, ktera bude drzet slovnik[trida]: columndatasource
# TODO: striknte oddelit frontend a backend

# TODO: moznost volby libovolneho algoritmu od scikitu

# TODO: @numba.njit()
# TODO: https://docs.bokeh.org/en/latest/docs/reference/core/templates.html?fbclid=IwAR1AzUzA2gmpmO3bGqtM7bpNrop-bzbHA3jjgP786VuJirgANO8m7Ia5qAk

# TODO: prepinac s nazvem souboru, kam se ulozi vysledek + prepinac po ukonceni aplikace ulozit dataset

# TODO: configurak v JSONu
# TODO: html stranky navic k proklikani a jako info

validation.silence(1002, True)  # silence bokeh plot warning


PALETTE = bokeh.palettes.Category10[10]


def parse_args():

    parser = argparse.ArgumentParser(description="bokeh plot")
    parser.add_argument('--dataset', default='', nargs=1)
    parser.add_argument('--cols', default=['x', 'y', 'classification'], nargs=3, help='x y classification')
    # parser.add_argument('--rcols', default='0 1', nargs=2)

    parsed = parser.parse_args(sys.argv[1:])

    return parsed.dataset, parsed.cols


if __name__ == '__main__':

    args = parse_args()
    if args[0] == '':
        df = dg.cluster_data_pandas(x_interval=(0, 30), y_interval=(-10, 10),
                                    clusters=CLUSTERS_COUNT_DEF, av_cluster_size=CLUSTER_SIZE_DEF,
                                    clust_size_vol=CLUSTER_VOL_DEF)  # TODO: hezceji
    else:
        path, column_names = args[0][0], args[1]
        df = read_df(path, column_names)


    def bkapp(doc):

        # if classification is None:
        #     classification = dg.classify(len(x_data), [str(i) for i in range(INIT_CLASSES_COUNT)])

        source_data = SourceData(df=df, palette=PALETTE)

        lay = BasicGeneralLayout(source_data=source_data)

        doc.add_root(row(lay.layout))

    server = Server({'/': bkapp}, num_procs=1)
    server.start()

    server.io_loop.add_callback(server.show, "/")
    try:
        server.io_loop.start()
    except KeyboardInterrupt:
        print()
        print("Session closed.")
