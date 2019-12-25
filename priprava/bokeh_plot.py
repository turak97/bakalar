import numpy as np
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, Slider, CustomJS
from bokeh.plotting import Figure, show
from bokeh.events import DoubleTap
from bokeh.io import curdoc
from bokeh.models import WheelZoomTool

from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import data_gen as dg

import operator


def polynomial_plots(x_data, y_data, min_degree, max_degree,
                     domain_ext=0.3):
    domain_extension = (x_data[-1] - x_data[0]) * domain_ext

    plots = {}
    for degree in range(min_degree, max_degree + 1):
        model_coef = dg.polynomial_model_coeff(degree, x_data, y_data)
        # how polynom really looks like for figure
        x_plot, y_plot = dg.polynom_line(model_coef,
                                         x_data[0] - domain_extension,
                                         x_data[-1] + domain_extension)

        plots[str(degree)] = y_plot
    return x_plot, plots


def polynomial_fig(source_data, min_degree=1, max_degree=10,
                   starting_degree=-1, x_ext=0.3, y_ext=0.3):
    if starting_degree < min_degree or starting_degree > max_degree:
        starting_degree = min_degree

    x_data = source_data.data['x']
    y_data = source_data.data['y']
    y_min, y_max = np.min(y_data), np.max(y_data)
    range_extension = (y_max - y_min) * y_ext
    x_plot, plots = polynomial_plots(x_data, y_data, min_degree, max_degree, domain_ext=x_ext)

    source_visible = ColumnDataSource(data=dict(
        x=x_plot, y=plots[str(starting_degree)]))
    source_available = ColumnDataSource(data=plots)

    wheel = WheelZoomTool(dimensions='width')
    plot = Figure(y_range=(y_min - range_extension, y_max + range_extension), active_scroll=wheel)
    plot.line('x', 'y', source=source_visible, line_width=3, line_alpha=0.6)
    plot.circle('x', 'y', source=source_data)

    slider = Slider(title='Trigonometric function',
                    value=int(starting_degree),
                    start=np.min([int(i) for i in plots.keys()]),
                    end=np.max([int(i) for i in plots.keys()]),
                    step=1)
    return plot, slider, source_visible, source_available


def bokeh_plot(x_data, y_data,
               x_ext=0.3, y_ext=0.3):

    source_data = ColumnDataSource(data=dict(x=x_data, y=y_data))
    plot, slider, source_visible, source_available = polynomial_fig(source_data, x_ext=x_ext, y_ext=y_ext)
    slider.callback = CustomJS(
        args=dict(source_visible=slider.visible,
                  source_available=source_available), code="""
            var selected_function = cb_obj.value;
            // Get the data from the data sources
            var data_visible = source_visible.data;
            var data_available = source_available.data;
            // Change y-axis data according to the selected value
            data_visible.y = data_available[selected_function];
            // Update the plot
            source_visible.change.emit();
        """)

    def callback(event):
        x_new, y_new = (event.x, event.y)
        x_arr_old = source_data.data['x']
        y_arr_old = source_data.data['y']
        x_arr_new, y_arr_new = dg.insert_point_x_sorted(x_arr_old, y_arr_old, x_new, y_new)
        # update source data plot layer
        source_data.data = dict(x=x_arr_new, y=y_arr_new)
        # plot.circle('x', 'y', source=source_data)

        x_plot, plots = polynomial_plots(x_arr_new, y_arr_new, 1, 10)
        source_visible.data = dict(x=x_plot, y=plots[str(slider.value)])  # update visible data in slider
        source_available.data = plots  # update available data in slider

    plot.on_event(DoubleTap, callback)
    layout = row(plot, slider)
    curdoc().add_root(layout)


data = dg.polynom_data(clusters=1)
bokeh_plot(data[0], data[1])

