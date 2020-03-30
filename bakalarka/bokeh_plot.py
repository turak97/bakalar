
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.io import curdoc
import bokeh

import data_gen as dg
import plotting_utilities as pu

import Layout as lo


# TODO: @numba.njit()
# TODO: pouzit patch pro update dat?

# TODO: u Widgetu menit vlastnosti pres update a ne primo
# TODO: zprovoznit vice moznosti generovani dat
# TODO: moznost nahrat vlastni CVScko, opakovane generovat data


POL_FROM_DGR = 1
POL_TO_DGR = 10

CLASSES_COUNT = 3

MESH_STEP_SIZE = 0.05  # detail of plotted picture

PALETTE = bokeh.palettes.Cividis[CLASSES_COUNT]


def bokeh_plot(x_data, y_data, classification=None,
               polynom_min_degree=POL_FROM_DGR, polynom_max_degree=POL_TO_DGR,
               x_ext=2, y_ext=0.3):

    if classification is None:
        classification = dg.classify(len(x_data), CLASSES_COUNT)

    plot_source = ColumnDataSource(
        data=dict(
            x=x_data.tolist(),
            y=y_data.tolist(),
            classification=classification.tolist(),
            color=[PALETTE[i] for i in classification]
        )
    )

    plot_info = pu.PlotInfo(plot_source=plot_source, pol_min_degree=POL_FROM_DGR,
                            pol_max_degree=POL_TO_DGR, palette=PALETTE,
                            x_extension=x_ext, y_extension=y_ext,
                            mesh_step_size=MESH_STEP_SIZE)

    data = lo.Data(x_data, y_data, classification)

    lay = lo.Layout(data=data, plot_source=plot_source, plot_info=plot_info)

    curdoc().add_root(row(lay.layout))


# data = dg.polynom_data(clusters=1, density=10, polynom=np.array([1 / 10, 1]), interval=(-50, 50))
init_data = dg.cluster_data(x_interval=(0, 30), y_interval=(-10, 10),
                       clusters=3, av_cluster_size=8, clust_size_vol=3)

bokeh_plot(init_data[0], init_data[1], classification=init_data[2])
