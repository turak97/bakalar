
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
from constants import POL_FROM_DGR, POL_TO_DGR, INIT_CLASSES_COUNT, X_EXT, Y_EXT
from in_n_out import read_df


# TODO: @numba.njit()
# TODO: pouzit patch pro update dat?

# TODO: u Widgetu menit vlastnosti pres update a ne primo
# TODO: zprovoznit vice moznosti generovani dat, moznost nahrat vlastni CVScko

# TODO: pekneji rozclenit plotting_utilities.py
# TODO: konstanty v extra souboru (nebo nejak standardizovane, jak se to v pytohnu dela)

validation.silence(1002, True)  # silence bokeh plot warning


PALETTE = bokeh.palettes.Category10[10]


def bokeh_plot(x_data, y_data, classification=None,
               polynom_min_degree=POL_FROM_DGR, polynom_max_degree=POL_TO_DGR,
               x_ext=2, y_ext=0.3):

    if classification is None:
        classification = dg.classify(len(x_data), [str(i) for i in range(INIT_CLASSES_COUNT)])

    plot_info = pu.PlotInfo(plot_source_init_data=(x_data.tolist(), y_data.tolist(), classification),
                            pol_min_degree=POL_FROM_DGR,
                            pol_max_degree=POL_TO_DGR, palette=PALETTE,
                            x_extension=x_ext, y_extension=y_ext)

    data = lo.Data(x_data, y_data, classification, len(set(classification)))

    lay = lo.Layout(data=data, plot_info=plot_info)

    curdoc().add_root(row(lay.layout))


# data = dg.polynom_data(clusters=1, density=10, polynom=np.array([1 / 10, 1]), interval=(-50, 50))

# init_data = dg.cluster_data(x_interval=(0, 30), y_interval=(-10, 10),
#                             clusters=INIT_CLASSES_COUNT, av_cluster_size=2, clust_size_vol=1)
#
# bokeh_plot(init_data[0], init_data[1], classification=init_data[2])

###################

# Setting num_procs here means we can't touch the IOLoop before now, we must
# let Server handle that. If you need to explicitly handle IOLoops then you
# will need to use the lower level BaseServer class.

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
                                    clusters=INIT_CLASSES_COUNT, av_cluster_size=2,
                                    clust_size_vol=1)  # TODO: hezceji
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
