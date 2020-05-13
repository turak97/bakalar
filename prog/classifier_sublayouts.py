
from bokeh.models import RadioButtonGroup, TextInput, Div, Select
from bokeh.layouts import row, column

from basic_sublayouts import ClassificationLike, BasicSubLayout, SliderSubLayout, NeuralSubLayout
from constants import POLY_DEF_DGR

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


class SvmClassification(BasicClassification):
    class ButtonStr:
        # kernel button
        LINEAR = "linear"
        POLY = "polynomial"
        RBF = "radial"
        SIGMOID = "sigmoid"

    def __init__(self, model_name, source_data):

        BasicClassification.__init__(self, model_name, source_data)

    def _init_button_layout(self):
        """Creates buttons bellow the figure, sets the trigger functions on them
        and add them to the subLayout"""
        total_width = 500

        __kernel_text = Div(text="Algorithm kernel: ")
        self.__kernel_button = RadioButtonGroup(
            labels=[self.ButtonStr.LINEAR, self.ButtonStr.POLY, self.ButtonStr.RBF, self.ButtonStr.SIGMOID],
            active=0, width=total_width
        )
        __kernel_group = column(__kernel_text, self.__kernel_button)

        self.__degree_button = Select(
            title="", value=str(POLY_DEF_DGR),
            options=[str(i) for i in range(20)], width=70)
        degree_text = Div(text="Degree (" + self.ButtonStr.POLY + "): ")

        self.__regularization_parameter_input = TextInput(value="1.0", width=75)
        regularization_parameter_text = Div(text="Regularization parameter: ")

        return column(
                      __kernel_group,
                      row(regularization_parameter_text,
                          self.__regularization_parameter_input),
                      row(degree_text, self.__degree_button)
                      )

    def _update_model_params(self):
        new_kernel = self.__label2kernel_str(
            self.__chosen_kernel()
        )
        self._model.kernel = new_kernel

        self._model.degree = int(self.__degree_button.value)  # __degree_button has predefined values

        self._model.C = self.__get_regularization()

    def __chosen_kernel(self):
        return self.__kernel_button.labels[self.__kernel_button.active]

    def __get_regularization(self):
        res = 1.0
        try:
            res = float(self.__regularization_parameter_input.value)
        except ValueError:
            self.__regularization_parameter_input.value = "1.0"
        return res

    @staticmethod
    def __label2kernel_str(label):
        if label == SvmClassification.ButtonStr.LINEAR:
            return "linear"
        elif label == SvmClassification.ButtonStr.POLY:
            return "poly"
        elif label == SvmClassification.ButtonStr.RBF:
            return "rbf"
        else:
            return "sigmoid"
