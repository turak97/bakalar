
from bokeh.layouts import row, column
from bokeh.models import Slider, ColumnDataSource

import numpy as np

from basic_sublayouts import RegressionSubLayout, SliderLike, NeuralLike

from math import ceil
from bokeh.models import PointDrawTool, Button, LassoSelectTool, Div, \
    CheckboxButtonGroup, Slider, TextInput, Toggle, RadioButtonGroup

from constants import POL_FROM_DGR, POL_TO_DGR, LOSS_PRINT, \
    NEURAL_DEF_SLIDER_STEPS, NEURAL_DEF_MAX_ITER_STEPS, NEURAL_DEF_LAYERS, NEURAL_DEF_ACTIVATION, NEURAL_DEF_SOLVER


class SliderRegressionSubLayout(SliderLike, RegressionSubLayout):
    def __init__(self, name, model, source_data, slider_params):
        # self._model_attr, slider_attr = slider_params
        # self._start, self._end, self._step, self._value = slider_attr

        SliderLike.__init__(self, slider_params)
        RegressionSubLayout.__init__(self, name, model, source_data)

        self._set_visible_renderer(1)

    def refit(self):
        self._info("Updating model and fitting data...")
        self._update_model_params()
        self._figure_update()
        self._set_visible_renderer(self._slider.value)
        self._info("Fitting and updating DONE")

    def _figure_update(self):
        self._info("Updating model and fitting data...")

        (x_min, x_max) = self.source_data.get_min_max_x()
        self._line_data = self.LineData(x_min, x_max, self._x_ext)

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
        self._line_data = self.LineData(x_min, x_max, self._x_ext)

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


class NeuralRegression(RegressionSubLayout):
    class ButtonStr:
        # activation button
        IDENTITY = "identity"
        SIGMOID = "logistic sigmoid"
        TANH = "tanh"
        LINEAR = "linear"
        # solver button
        LBFGS = "lbfgs"
        GRADIENT = "gradient descent"
        ADAM = "adam"

    def __init__(self, name, model, source_data):
        """Creates attribute self.name, self.classifier, self.fig, self.layout self.source_data from super"""
        # initialise iteration parameters for slider and classifier fitting
        self._neural_data = []

        self.__update_iteration_params(NEURAL_DEF_MAX_ITER_STEPS, NEURAL_DEF_SLIDER_STEPS)
        self.__logarithmic_steps = False

        RegressionSubLayout.__init__(self, name, model, source_data)
        self.__set_visible_renderer(self.__slider_steps)

    def refit(self):
        """Update iteration (max, slider step, ...) and classifier parameters then update figure"""
        self._info("Updating model and fitting data...")
        self.__logarithmic_steps = self.__logarithm_button.active

        self.__update_iteration_params(int(self.__max_iterations_input.value),
                                       int(self.__slider_steps_input.value))
        self._update_model_params()

        self._figure_update()
        self.__set_visible_renderer(self.__slider_steps)

        self._info("Fitting and updating DONE")

    def _figure_update(self):
        self._info("Updating model and fitting data...")

        (x_min, x_max) = self.source_data.get_min_max_x()
        self._line_data = self.LineData(x_min, x_max, self._x_ext)
        self._neural_data = []

        self._model.max_iter = self.__iter_step
        prev = 0  # used only with logarithmic slider
        for iterations, renderer_i in zip(range(self.__iter_step, self.__max_iter_steps + 1,
                                                self.__iter_step),
                                          range(1, self.__slider_steps + 1)):  # first one is Circle
            if self.__logarithmic_steps:
                """
                in fact it is not logarithmic (I chose this name because I find it rather intuitive).
                This option allows user to see the begging of the learning
                process when the changes are much more significant in more detail.
                For  5000 iterations max and 10 steps it will be:
                50, 111, 187, 285, 416, 600, 875, 1333, 2250, 5000
                """
                log_iter_total = int(iterations / (self.__slider_steps - renderer_i + 1))
                self._model.max_iter = max(log_iter_total - prev,
                                           1)
                print(self._model.max_iter)
                prev = log_iter_total

            self._fit_and_render(renderer_i)

        self._info("Done")

    def _fit_and_render(self, renderer_i):
        """Fits the model, render image and add/update image to the figure
        expects attribute self.__img_data
        """
        self._info("Fitting data and updating figure, step: " + str(renderer_i))

        x_data, y_data = self.source_data.data_to_regression_fit()
        self._model.fit(x_data, y_data)
        print(self._model.max_iter)

        self._neural_data.append(self._model.loss_)

        y_line = self._model.predict(np.c_[self._line_data.xx.ravel()])
        self._line_data.add_line(y_line)

        if len(self._fig.renderers) - 1 < renderer_i:
            print("adding renderer no " + str(renderer_i - 1))
            self._new_fig_renderer(renderer_i - 1)
        else:
            print("updating render no " + str(renderer_i - 1))
            self._update_fig_renderer(renderer_i)

    def _init_button_layout(self):
        """Creates buttons bellow the figure, sets the trigger functions on them
        and add them to the subLayout."""
        total_width = 500

        if LOSS_PRINT == "app":
            self._loss_info = Div(text="Validation loss: ")
            self._loss_text = Div(text="")
            loss_group = row(self._loss_info, self._loss_text)
        else:
            loss_group = row()

        self.__iteration_slider = Slider(start=self.__iter_step, end=self.__max_iter_steps,
                                         step=self.__iter_step, value=self.__max_iter_steps,
                                         title="Iterations", width=total_width)
        self.__iteration_slider.on_change('value', self.__slider_change)
        max_iteration_text = Div(text="Max iterations:")
        self.__max_iterations_input = TextInput(value=str(self.__max_iter_steps), width=63)
        number_step_text = Div(text="Slider steps:")
        self.__slider_steps_input = TextInput(value=str(self.__slider_steps), width=45)
        self.__logarithm_button = Toggle(label="Logarithmic slider", width=55)
        slider_group = column(self.__iteration_slider,
                              row(max_iteration_text, self.__max_iterations_input,
                                  number_step_text, self.__slider_steps_input,
                                  self.__logarithm_button)
                              )

        self.__layers_input = TextInput(value=NEURAL_DEF_LAYERS)
        layers_text = Div(text="Hidden layers sizes:")
        layers_input = column(layers_text, row(self.__layers_input))

        self.__activation_button = RadioButtonGroup(
            labels=[self.ButtonStr.IDENTITY, self.ButtonStr.SIGMOID,
                    self.ButtonStr.TANH, self.ButtonStr.LINEAR], active=NEURAL_DEF_ACTIVATION,
            width=total_width
        )
        activation_text = Div(text="Activation function in hidden layers:")
        activation_group = column(activation_text, self.__activation_button)

        self.__solver_button = RadioButtonGroup(
            labels=[self.ButtonStr.LBFGS, self.ButtonStr.GRADIENT, self.ButtonStr.ADAM],
            active=NEURAL_DEF_SOLVER,
            width=total_width
        )
        solver_text = Div(text="Weigh optimization solver:")
        solver_group = column(solver_text, self.__solver_button)

        return column(loss_group,
                      slider_group,
                      layers_input, activation_group,
                      solver_group)

    def __set_visible_renderer(self, visible):
        print(visible)
        for renderer, i in zip(self._fig.renderers[1:], range(1, len(self._fig.renderers))):
            if i == visible:
                renderer.visible = True
                if LOSS_PRINT == 'app':
                    self._loss_text.update(text=str(self._neural_data[i - 1]))
                elif LOSS_PRINT == 'log':
                    self._info("Validation loss: " + str(self._neural_data[i - 1]))
            else:
                renderer.visible = False

        if self.__logarithmic_steps:
            self.__iteration_slider.show_value = False
            self.__iteration_slider.title = "Iterations logarithmic: " + str(
                int(self.__iteration_slider.value / (self.__slider_steps - visible + 1)))
        else:
            self.__iteration_slider.show_value = True
            self.__iteration_slider.title = "Iterations"

    def __slider_change(self, attr, old, new):
        visible = int(self.__iteration_slider.value / self.__iter_step)
        self.__set_visible_renderer(visible)

    def __update_iteration_params(self, max_iter_steps, slider_steps):
        """Update iteration parameters
        and if __iteration_slider is initialised (typically after first fit), update slider parameters.
        """

        self.__iter_step = ceil(max_iter_steps/slider_steps)
        self.__max_iter_steps = self.__iter_step * slider_steps
        self.__slider_steps = slider_steps
        try:
            self.__iteration_slider.update(
                start=self.__iter_step,
                end=self.__max_iter_steps,
                step=self.__iter_step,
                value=self.__max_iter_steps
            )
        except AttributeError:
            pass

    def _update_model_params(self):
        new_activation = self.__label2activation_str(
            self.__activation_button.labels[self.__activation_button.active]
        )
        self._model.activation = new_activation

        new_solver = self.__label2solver_str(
            self.__solver_button.labels[self.__solver_button.active]
        )
        self._model.solver = new_solver

        self._model.hidden_layer_sizes = self.__text2layers(self.__layers_input.value)

    @staticmethod
    def __text2layers(layers_str):
        return tuple([int(i) for i in layers_str.split(",")])

    @staticmethod
    def __label2activation_str(label):
        """transform string from button to string that classifier expects"""
        if label == NeuralRegression.ButtonStr.IDENTITY:
            return "identity"
        elif label == NeuralRegression.ButtonStr.SIGMOID:
            return "logistic"
        elif label == NeuralRegression.ButtonStr.TANH:
            return "tanh"
        else:
            return "relu"

    @staticmethod
    def __label2solver_str(label):
        if label == NeuralRegression.ButtonStr.LBFGS:
            return "lbfgs"
        elif label == NeuralRegression.ButtonStr.GRADIENT:
            return "sgd"
        else:
            return "adam"
