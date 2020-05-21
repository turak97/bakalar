
from generic_sublayouts import ClassificationLike, BasicSubLayout, SliderSubLayout, NeuralSubLayout, SvmSubLayout

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


class BasicClassification(ClassificationLike, BasicSubLayout):
    def __init__(self, model_name, source_data):
        ClassificationLike.__init__(self, model_name, source_data)
        BasicSubLayout.__init__(self, model_name)

        self.refit()


class SliderClassification(ClassificationLike, SliderSubLayout):
    def __init__(self, model_name, source_data, slider_parameters):
        ClassificationLike.__init__(self, model_name, source_data)
        SliderSubLayout.__init__(self, model_name, slider_parameters)

        self.refit()


class NeuralClassification(ClassificationLike, NeuralSubLayout):
    def __init__(self, model_name, source_data):
        ClassificationLike.__init__(self, model_name, source_data)
        NeuralSubLayout.__init__(self, model_name)

        self.refit()


class SvmClassification(ClassificationLike, SvmSubLayout):
    def __init__(self, model_name, source_data):
        ClassificationLike.__init__(self, model_name, source_data)
        SvmSubLayout.__init__(self, model_name)

        self.refit()

