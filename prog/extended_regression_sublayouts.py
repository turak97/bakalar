
from bokeh.layouts import row, column
from bokeh.models import Slider, ColumnDataSource

from basic_sublayouts import RegressionSubLayout, SliderLike

from constants import POL_FROM_DGR, POL_TO_DGR


class SliderRegressionSubLayout(SliderLike, RegressionSubLayout):
    def __init__(self, name, model, source_data, slider_params):
        # self._model_attr, slider_attr = slider_params
        # self._start, self._end, self._step, self._value = slider_attr

        SliderLike.__init__(self, slider_params)
        RegressionSubLayout.__init__(self, name, model, source_data)

    def refit(self):
        self._info("Updating model and fitting data...")
        self._update_model_params()
        self._figure_update()
        self._set_visible_renderer(self._slider.value)
        self._info("Fitting and updating DONE")

    def _figure_update(self):
        self._info("Updating model and fitting data...")

        (x_min, x_max) = self.source_data.get_min_max_x()
        self._rend_data = self.LineData(x_min, x_max, self._x_ext)

        for value, i in zip(range(self._start, self._end + 1, self._step),
                            range(1, self._end + 1, self._step)):
            setattr(self._model, self._model_attr, value)

            self._fit_and_render(i)

        self._info("Done")

    def _set_visible_renderer(self, visible):
        for renderer, i in zip(self._fig.renderers[1:], range(1, len(self._fig.renderers))):
            if i == visible:
                renderer.visible = True
            else:
                renderer.visible = False


class PolynomialRegression(SliderRegressionSubLayout):
    def __init__(self, name, model, source_data):

        self._pol_from_degree, self._pol_to_degree = POL_FROM_DGR, POL_TO_DGR

        RegressionSubLayout.__init__(self, name, model, source_data)

    def refit(self):
        self._info("Updating model and fitting data...")
        self._update_model_params()
        self._figure_update()
        self.__set_visible_renderer(self.__slider.value)
        self._info("Fitting and updating DONE")

    def _figure_update(self):

        self._info("Updating model and fitting data...")

        (x_min, x_max) = self.source_data.get_min_max_x()
        self._rend_data = self.LineData(x_min, x_max, self._x_ext)

        for degree, i in zip(range(self._pol_from_degree, self._pol_to_degree + 1),
                             range(1, self._pol_to_degree + 1)):
            self._model.set_params(poly__degree=degree)

            self._fit_and_render(i)

        self._info("Done")

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
