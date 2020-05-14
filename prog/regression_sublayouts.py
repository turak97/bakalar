
from basic_sublayouts import RegressionLike, BasicSubLayout, SliderSubLayout, NeuralSubLayout

from constants import POL_TO_DGR, POL_FROM_DGR, POLY_DEF_DGR


class BasicRegression(RegressionLike, BasicSubLayout):
    def __init__(self, model_name, source_data):
        RegressionLike.__init__(self, model_name, source_data)
        BasicSubLayout.__init__(self, model_name)

        self.refit()


class PolynomialRegression(RegressionLike, SliderSubLayout):
    def __init__(self, model_name, source_data):
        RegressionLike.__init__(self, model_name, source_data)
        slider_attr = (POL_FROM_DGR, POL_TO_DGR, 1, POLY_DEF_DGR)
        slider_params = ("Polynomial degree", slider_attr)
        SliderSubLayout.__init__(self, model_name, slider_params)

        self.refit()

    def _figure_update(self):
        self._info("Initialising model and render data...")
        self._init_data()
        self._init_model()

        for value, i in zip(range(self._start, self._end + 1, self._step),
                            range(1, self._end + 1, self._step)):
            self._info("Setting model attribute: " + str(self._model_attr) + " to value " + str(value))
            self._model.set_params(poly__degree=value)

            self._info("Training model...")
            self._fit()
            self._info("Adding renderer number " + str(i) + "...")
            self._render(self._fig, i)

        visible = int((self._slider.value - self._start)/self._step) + 1
        self._set_visible_renderer(visible)


class SliderRegression(RegressionLike, SliderSubLayout):
    def __init__(self, model_name, source_data, slider_parameters):
        RegressionLike.__init__(self, model_name, source_data)
        SliderSubLayout.__init__(self, model_name, slider_parameters)

        self.refit()


class NeuralRegression(RegressionLike, NeuralSubLayout):
    def __init__(self, model_name, source_data):
        RegressionLike.__init__(self, model_name, source_data)
        NeuralSubLayout.__init__(self, model_name)

        self.refit()
