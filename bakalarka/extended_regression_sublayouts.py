
from bokeh.layouts import row, column
from bokeh.models import Slider, ColumnDataSource

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import numpy as np

from basic_sublayouts import RegressionSubLayout

from constants import POL_FROM_DGR, POL_TO_DGR

# TODO: pekneji rozclenit refit


class PolynomialRegression(RegressionSubLayout):
    def __init__(self, name, source_data):
        model = None
        self._pol_from_degree, self._pol_to_degree = POL_FROM_DGR, POL_TO_DGR

        RegressionSubLayout.__init__(self, name, model, source_data)

    def _init_button_layout(self):
        self.__slider = Slider(
            title='Polynom degree',
            value=POL_FROM_DGR, start=POL_FROM_DGR, end=POL_TO_DGR,
            step=1)
        self.__slider.on_change('value', self.__slider_change)
        return column(self.__slider)

    def __slider_change(self, attr, old, new):
        visible = new
        self.__set_visible_renderer(visible)

    def __set_visible_renderer(self, visible):
        for renderer, i in zip(self._fig.renderers[1:], range(1, len(self._fig.renderers))):
            if i == visible:
                renderer.visible = True
            else:
                renderer.visible = False

    def refit(self):
        x_min, x_max = self.source_data.get_min_max_x()
        x_range_extension = (x_max - x_min) * self._x_ext

        sources = []
        i = 0
        for degree in range(self._pol_from_degree, self._pol_to_degree + 1):
            model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                              ('linear', LinearRegression(fit_intercept=False))])

            x_to_fit, y_to_fit = self.source_data.data_to_regression_fit()
            model_fit = model.fit(x_to_fit, y_to_fit)
            model_coeff = model_fit.named_steps['linear'].coef_

            x_plot, y_plot = self.__polynom_line(model_coeff,
                                                 x_min - x_range_extension,
                                                 x_max + x_range_extension)

            source = ColumnDataSource(
                data=dict(
                    x=x_plot,
                    y=y_plot
                )
            )
            if len(self._fig.renderers) - 1 < i:
                self._fig.line(x='x', y='y', source=source)
            else:
                pass

            i += 1

        self.__set_visible_renderer(1)

    def __polynom_line(self, model_coef, x_from, x_to, steps=1000):
        model_coef = model_coef[::-1]  # numpy poly1d expects coefficients in reversed order
        f = np.poly1d(model_coef)
        x_plot = np.linspace(x_from, x_to, steps)
        y_plot = f(x_plot)
        return x_plot, y_plot

