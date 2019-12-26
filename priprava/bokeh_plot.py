import numpy as np
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, Slider, CustomJS
from bokeh.plotting import Figure, show
from bokeh.events import DoubleTap
from bokeh.io import curdoc
from bokeh.models import WheelZoomTool
from bokeh.models import PointDrawTool

from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import data_gen as dg

import operator


def polynomial_plots(x_data, y_data, min_degree, max_degree,
                     domain_ext=0.3):
    x_min, x_max = np.min(x_data), np.max(x_data)
    domain_extension = (x_max - x_min) * domain_ext

    plots = {}
    for degree in range(min_degree, max_degree + 1):
        model_coef = dg.polynomial_model_coeff(degree, x_data, y_data)
        # how polynom really looks like for figure
        x_plot, y_plot = dg.polynom_line(model_coef,
                                         x_min - domain_extension,
                                         x_max + domain_extension)

        plots[str(degree)] = y_plot
    return x_plot, plots


def polynomial_fig(x_data, y_data, min_degree=1, max_degree=10,
                   starting_degree=-1, x_ext=0.3, y_ext=0.3):
    if starting_degree < min_degree or starting_degree > max_degree:
        starting_degree = min_degree

    y_min, y_max = np.min(y_data), np.max(y_data)
    range_extension = (y_max - y_min) * y_ext
    x_plot, plots = polynomial_plots(x_data, y_data, min_degree, max_degree, domain_ext=x_ext)

    source_visible = ColumnDataSource(data=dict(
        x=x_plot, y=plots[str(starting_degree)]))
    source_available = ColumnDataSource(data=plots)

    plot = Figure(y_range=(y_min - range_extension, y_max + range_extension))
    plot.line('x', 'y', source=source_visible, line_width=3, line_alpha=0.6)
    slider = Slider(title='Polynom degree',
                    value=int(starting_degree),
                    start=np.min([int(i) for i in plots.keys()]),
                    end=np.max([int(i) for i in plots.keys()]),
                    step=1)
    return plot, slider, source_visible, source_available


def bokeh_plot(x_data, y_data,
               x_ext=2, y_ext=0.3):

    plot, slider, source_visible, source_available = polynomial_fig(
        x_data, y_data, x_ext=x_ext, y_ext=y_ext)

    circle_source = ColumnDataSource(data=dict(x=x_data.tolist(), y=y_data.tolist()))
    move_circle = plot.circle('x', 'y', source=circle_source, size=7)

    tool = PointDrawTool(renderers=[move_circle], empty_value='added', add=True)
    slider.callback = CustomJS(
        args=dict(source_visible=source_visible,
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

    def data_change(attr, old, new):
        x_arr_new, y_arr_new = np.array(new['x']), np.array(new['y'])

        x_plot, plots = polynomial_plots(x_arr_new, y_arr_new, 1, 10, x_ext)
        source_visible.data = dict(x=x_plot, y=plots[str(slider.value)])  # update visible data in slider
        source_available.data = plots  # update available data in slider
    circle_source.on_change('data', data_change)

    plot.add_tools(tool)
    layout = row(plot, slider)
    curdoc().add_root(layout)


data = dg.polynom_data(clusters=5, density=10, polynom=np.array([1/10, 1]), interval=(-1000, 1000))
bokeh_plot(data[0], data[1])

