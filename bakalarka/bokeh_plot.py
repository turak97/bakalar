
import argparse, sys
from pathlib import Path

from bokeh.layouts import row, column
from bokeh.io import curdoc


import bokeh
from bokeh.server.server import  Server
from bokeh.core import validation

import data_gen as dg
import plotting_utilities as pu
import Layout as lo
from constants import POL_FROM_DGR, POL_TO_DGR, X_EXT, Y_EXT
from constants import CLUSTER_SIZE_DEF, CLUSTER_VOL_DEF, CLUSTERS_COUNT_DEF
from in_n_out import read_df


# TODO: @numba.njit()
# TODO: pouzit patch pro update dat?

# TODO: u Widgetu menit vlastnosti pres update a ne primo
# TODO: zprovoznit vice moznosti generovani dat, moznost nahrat vlastni CVScko

# TODO: pekneji rozclenit plotting_utilities.py
# TODO: konstanty v extra souboru (nebo nejak standardizovane, jak se to v pytohnu dela)

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

        plot_info = pu.PlotInfo(df=df,
                                pol_min_degree=POL_FROM_DGR,
                                pol_max_degree=POL_TO_DGR, palette=PALETTE,
                                x_extension=X_EXT, y_extension=Y_EXT)

        data = lo.Data(df['x'], df['y'], df['classification'], len(set(df['classification'])))

        lay = lo.Layout(data=data, plot_info=plot_info)

        doc.add_root(row(lay.layout))

    server = Server({'/': bkapp}, num_procs=1)
    server.start()

    server.io_loop.add_callback(server.show, "/")
    try:
        server.io_loop.start()
    except KeyboardInterrupt:
        print()
        print("Session closed.")
