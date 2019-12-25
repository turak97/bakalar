from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Column
from bokeh.io import curdoc
from bokeh.events import DoubleTap
from bokeh.plotting import Figure, show
from bokeh.server.server import Server
from tornado.ioloop import IOLoop


import data_gen as dg

import numpy as np

RANGE = 500
coordList = []
TOOLS = "tap"

def make_plot(x, y):
    p = figure(title='Double click to leave a dot.',
               tools=TOOLS,
               width=700, height=700,
               x_range=(-RANGE, RANGE), y_range=(np.min(y) - 1000, np.max(y) + 1000))

    source = ColumnDataSource(data=dict(x=x, y=y))
    p.circle(source=source, x='x', y='y')
    return p, source

def do_stuff():
    data = dg.polynom_data(clusters=1, density=5)
    x = data[0]
    y = data[1]

    p, source = make_plot(x, y)

    # add a dot where the click happened
    def callback(event):
        x_new, y_new = (event.x, event.y)
        x_arr_old = source.data['x']
        y_arr_old = source.data['y']
        x_arr_new, y_arr_new = dg.insert_point_x_sorted(x_arr_old, y_arr_old, x_new, y_new)
        source.data = dict(x=x_arr_new, y=y_arr_new)

    p.on_event(DoubleTap, callback)
    layout = Column(p)
    curdoc().add_root(layout)

do_stuff()

