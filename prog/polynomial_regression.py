import numpy as np
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Slider, PointDrawTool
from bokeh.models.callbacks import CustomJS
from bokeh.plotting import figure

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import data_gen as dg
from generic_sublayouts import SubLayout


# TODO: BUG - linearni regrese pro vice y bodu na jediny x bod


# fit model and return coefficients
def polynomial_model_coeff(degree, x_data, y_data):
    model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                      ('linear', LinearRegression(fit_intercept=False))])
    # print(x_data)
    # print(y_data)
    model = model.fit(x_data[:, np.newaxis], y_data)
    return model.named_steps['linear'].coef_


def polynomial_plots(x_data, y_data, min_degree, max_degree,
                     domain_ext=0.3):
    x_min, x_max = np.min(x_data), np.max(x_data)
    domain_extension = (x_max - x_min) * domain_ext

    y_plots = {}
    for degree in range(min_degree, max_degree + 1):
        # fit model and return coefficients
        model_coef = polynomial_model_coeff(degree, x_data, y_data)
        # make data for curve render
        x_plot, y_plot = dg.polynom_line(model_coef,
                                         x_min - domain_extension,
                                         x_max + domain_extension)

        y_plots[str(degree)] = y_plot
    return x_plot, y_plots


# Creates figure and data for polynomial regression plot
def polynomial_line_sources(x_data, y_data, min_degree, max_degree,
                   starting_degree, x_ext, y_ext):

    x_plot, y_plots = polynomial_plots(x_data, y_data, min_degree, max_degree, domain_ext=x_ext)

    # create sources from data
    # source_visible is a source with x and y values for current display
    # source_available are all y values (there are not x values)
    source_visible = ColumnDataSource(data=dict(
        x=x_plot, y=y_plots[str(starting_degree)]))
    source_available = ColumnDataSource(data=y_plots)

    return source_visible, source_available


def polynomial_figure(x_from, x_to, source):
    # create figure
    x_range = (x_from, x_to)
    y_range = x_range  # figure scope should be square
    fig = figure(y_range=y_range, x_range=x_range)
    fig.line('x', 'y', source=source, line_width=3, line_alpha=0.6)
    return fig


# function for update data in polynomial layout sources
# this updates data in sources and does not create a new ones like function polynomial_sources
def polynomial_layout_update(line_source_visible, line_sources_available,
                             x_data_new, y_data_new,
                             min_degree, max_degree, slider_value, x_ext):
    x_plot, y_plots = polynomial_plots(x_data_new, y_data_new, min_degree, max_degree, x_ext)
    line_source_visible.data = dict(x=x_plot, y=y_plots[str(slider_value)])  # update currently visible data in slider
    line_sources_available.data = y_plots  # update available data in slider


# create a figure and slider for plotting polynomial regression
def polynomial_layout(name, data, plot_info):

    # create source for figure and slider
    line_source_visible, line_sources_available = polynomial_line_sources(
        x_data=data.x_data, y_data=data.y_data,
        min_degree=plot_info.pol_min_degree, max_degree=plot_info.pol_max_degree,
        starting_degree=plot_info.pol_min_degree,
        x_ext=plot_info.x_extension, y_ext=plot_info.y_extension
    )

    # create figure
    x_min, x_max = np.min(data.x_data), np.max(data.x_data)
    x_range_extension = (x_max - x_min) * plot_info.x_extension
    fig = polynomial_figure(x_from=x_min - x_range_extension,
                            x_to=x_max + x_range_extension,
                            source=line_source_visible)
    # add line of polynom
    fig.line('x', 'y', source=line_source_visible, line_width=2)

    # add original data to the figure and prepare PointDrawTool to make them interactive
    move_circle = fig.circle('x', 'y', source=plot_info.plot_source, size=7)
    point_draw_tool = PointDrawTool(renderers=[move_circle], empty_value='black', add=True)
    fig.add_tools(point_draw_tool)

    # create slider
    slider = Slider(title='Polynom degree',
                    value=plot_info.pol_min_degree,
                    start=plot_info.pol_min_degree,
                    end=plot_info.pol_max_degree,
                    step=1)

    # make slider interactive with figure
    js_callback = CustomJS(  # TODO: upravit/zjistit zdroj pro citaci
        args=dict(source_visible=line_source_visible,
                  source_available=line_sources_available), code="""
            var selected_function = cb_obj.value;
            // Get the data from the data sources
            var data_visible = source_visible.data;
            var data_available = source_available.data;
            // Change y-axis data according to the selected value
            data_visible.y = data_available[selected_function];
            // Update the plot
            source_visible.change.emit();
        """)
    slider.js_on_change('value', js_callback)

    pol_lay = PolynomialLayout(name=name, fig=fig, slider=slider,
                               line_source_visible=line_source_visible,
                               line_sources_available=line_sources_available)

    return pol_lay


class PolynomialLayout(SubLayout):
    def __init__(self, name, fig, slider,
                 line_source_visible, line_sources_available,):
        self.name = name
        self.layout = column(slider, fig)
        self.fig = fig
        self.slider = slider
        self.line_source_visible = line_source_visible
        self.line_sources_available = line_sources_available

    def data_update(self, data, plot_info):
        polynomial_layout_update(
            line_source_visible=self.line_source_visible,
            line_sources_available=self.line_sources_available,
            x_data_new=data.x_data, y_data_new=data.y_data,
            slider_value=self.slider.value,
            min_degree=plot_info.pol_min_degree, max_degree=plot_info.pol_max_degree,
            x_ext=plot_info.x_extension)

