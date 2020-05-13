
from basic_sublayouts import RegressionLike, BasicSubLayout, SliderSubLayout, NeuralSubLayout


class BasicRegression(RegressionLike, BasicSubLayout):
    def __init__(self, model_name, source_data):
        RegressionLike.__init__(self, model_name, source_data)
        BasicSubLayout.__init__(self, model_name)

        self.refit()


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
