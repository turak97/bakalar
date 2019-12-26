from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Column
from bokeh.io import curdoc
from bokeh.events import DoubleTap
from bokeh.plotting import Figure, show
from bokeh.models import PointDrawTool
from bokeh.events import DoubleTap, Tap, PressUp, PinchEnd, PanEnd
import pandas as pd

import data_gen as dg

import numpy as np

RANGE = 500
coordList = []

def make_plot(x, y):
    p = figure(title='Double click to leave a dot.',
               width=700, height=700,
               x_range=(-RANGE, RANGE), y_range=(np.min(y) - 1000, np.max(y) + 1000))

    source = ColumnDataSource(data=dict(x=x, y=y))
    c1 = p.circle(source=source, x='x', y='y', radius=5)
    return c1, p, source


def do_stuff():
    data = dg.polynom_data(clusters=1, density=5)
    x = data[0]
    y = data[1]

    p = figure(title='Double click to leave a dot.',
               width=700, height=700,
               x_range=(-RANGE, RANGE), y_range=(np.min(y) - 1000, np.max(y) + 1000))

    x_list = x.tolist()
    y_list = y.tolist()

    source = ColumnDataSource(data=dict(x=x_list, y=y_list))

    c1 = p.circle(source=source, x='x', y='y', size=10)

    model_coef = dg.polynomial_model_coeff(3, x, y)
    x_line, y_line = dg.polynom_line(model_coef, -500, 500)
    line_source = ColumnDataSource(data=dict(x=x_line, y=y_line))
    p.line('x', 'y', source=line_source)

    tool = PointDrawTool(renderers=[c1], empty_value='added', add=True)

    def data_act(attr, old, new):
        np_x = np.array(new['x'])
        np_y = np.array(new['y'])

        model_coef = dg.polynomial_model_coeff(3, np_x, np_y)
        x_line, y_line = dg.polynom_line(model_coef, -500, 500)
        line_source.data = dict(x=x_line, y=y_line)

    source.on_change('data', data_act)


    # # add a dot where the click happened
    # def callback(event):
    #     x_new, y_new = (event.x, event.y)
    #     x_arr_old = source.data['x']
    #     y_arr_old = source.data['y']
    #     x_arr_new, y_arr_new = dg.insert_point_x_sorted(x_arr_old, y_arr_old, x_new, y_new)
    #     source.data = dict(x=x_arr_new, y=y_arr_new)
    #
    # p.on_event(DoubleTap, callback)

    layout = Column(p)
    p.toolbar.active_tap = tool
    p.add_tools(tool)

    curdoc().add_root(layout)

do_stuff()

