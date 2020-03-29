
from ClassifierLayout import ClassifierLayout, ImageData


from bokeh.models import RadioButtonGroup, TextInput, Button, Div, Slider, Toggle
from sklearn.neural_network import MLPClassifier
from bokeh.layouts import row, column
from math import ceil

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

STARTING_SLIDER_STEPS = 5
STARTING_MAX_ITER_STEPS = 200

import time


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


class NeuralClassifierLayout(ClassifierLayout):
    def __init__(self, name, data, plot_info):
        """
        creates attribute self.name, self.classifier, self.fig, self.layout
        self.data, self.plot_info from super
        """
        classifier = MLPClassifier(activation='tanh', hidden_layer_sizes=(10, 10), random_state=1,
                                   max_iter=200, tol=0)  # max_iter is irrelevant at this point
        # initialise iteration parameters for slider and classifier fitting
        self.__update_iteration_params(STARTING_MAX_ITER_STEPS, STARTING_SLIDER_STEPS)
        self.__logarithmic_steps = False

        ClassifierLayout.__init__(self, name, classifier, data, plot_info)
        self.__set_visible_renderer(self.slider_steps)

    def figure_update(self):
        self._info("Updating model and fitting data...")

        img_data = ImageData(self.data.x_data.min() - 1, self.data.x_data.max() + 1,
                             self.data.y_data.min() - 1, self.data.y_data.max() + 1,
                             self.plot_info.mesh_step_size)
        for iterations, renderer_i in zip(range(self.iter_step, self.max_iter_steps + 1,
                                                self.iter_step),
                                          range(1, self.slider_steps + 1)):  # first one is Circle
            if self.__logarithmic_steps:
                """
                in fact it is not logarithmic (I chose this name because I find it intuitive).
                This option allows user to see in more clearly the begging of the learning
                process when the changes are much more significant.
                For  5000 iterations max and 10 steps it will be:
                50, 111, 187, 285, 416, 600, 875, 1333, 2250, 5000
                """
                self.classifier.max_iter = int(iterations / (self.slider_steps - renderer_i + 1))
            else:
                self.classifier.max_iter = iterations
            self._fit_and_render(img_data, renderer_i)

        self._info("Done")

    def __set_visible_renderer(self, visible):
        for renderer, i in zip(self.fig.renderers[1:], range(1, len(self.fig.renderers))):
            if i == visible:
                renderer.visible = True
            else:
                renderer.visible = False

        if self.__logarithmic_steps:
            self.iteration_slider.title = "Iterations logarithmic: " + str(
                int(self.iteration_slider.value / (self.slider_steps - visible + 1))) + " ... "

    def __activation_change(self, attr, old, new):
        new_activation = self.__label2activation_str(
            self.activation_button.labels[new]
        )
        self.classifier.activation = new_activation
        self.figure_update()

    def __solver_change(self, attr, old, new):
        new_solver = self.__label2solver_str(
            self.solver_button.labels[new]
        )
        self.classifier.solver = new_solver
        self.figure_update()

    def __refit(self):
        self.fit_button.disabled = True  # disabling button so there are peaceful conditions for fitting model
        self.__logarithmic_steps = self.logarithm_button.active
        if not self.__logarithmic_steps:
            self.iteration_slider.title = "Iterations"

        self.__update_iteration_params(int(self.max_iterations_input.value),
                                       int(self.slider_steps_input.value))

        new_layers = self.__text2layers(self.layers_input.value)
        self.classifier.hidden_layer_sizes = new_layers


        self.figure_update()
        self.__set_visible_renderer(self.slider_steps)

        self.fit_button.disabled = False

    def __slider_change(self, attr, old, new):
        visible = int(self.iteration_slider.value/self.iter_step)
        self.__set_visible_renderer(visible)

    def _init_button_layout(self):
        """creates buttons bellow the figure and sets the trigger functions on them"""
        total_width = 500
        self.iteration_slider = Slider(start=self.iter_step, end=self.max_iter_steps,
                                       step=self.iter_step, value=self.max_iter_steps,
                                       title="Iterations", width=total_width)
        self.iteration_slider.on_change('value', self.__slider_change)
        max_iteration_text = Div(text="Max iterations:")
        self.max_iterations_input = TextInput(value=str(self.max_iter_steps), width=63)
        number_step_text = Div(text="Slider steps:")
        self.slider_steps_input = TextInput(value=str(self.slider_steps), width=45)
        self.logarithm_button = Toggle(label="Logarithmic slider", width=55)
        slider_group = column(self.iteration_slider,
                              row(max_iteration_text, self.max_iterations_input,
                                  number_step_text, self.slider_steps_input,
                                  self.logarithm_button)
                              )

        self.layers_input = TextInput(value="10, 10")
        text_text = Div(text="Hidden layers size:")
        layers_input = column(text_text, row(self.layers_input))

        self.activation_button = RadioButtonGroup(
            labels=[ButtonStr.IDENTITY, ButtonStr.SIGMOID,
                    ButtonStr.TANH, ButtonStr.LINEAR], active=3,
            width=total_width
        )
        self.activation_button.on_change('active', self.__activation_change)
        activation_text = Div(text="Activation function in hidden layers:")
        activation_group = column(activation_text, self.activation_button)

        self.solver_button = RadioButtonGroup(
            labels=[ButtonStr.LBFGS, ButtonStr.GRADIENT,
                    ButtonStr.ADAM], active=2,
            width=total_width
        )
        self.solver_button.on_change('active', self.__solver_change)
        solver_text = Div(text="Weigh optimization solver:")
        solver_group = column(solver_text, self.solver_button)

        self.fit_button = Button(label="Fit", button_type="success")
        self.fit_button.on_click(self.__refit)

        self.layout.children[2] = column(self.fit_button, slider_group,
                                         layers_input, activation_group,
                                         solver_group)

    def __update_iteration_params(self, max_iter_steps, slider_steps):
        # this sequence prevents unwanted behaviour when max_iter_steps/slider_steps needs to be rounded
        self.iter_step = ceil(max_iter_steps/slider_steps)
        self.max_iter_steps = self.iter_step * slider_steps
        self.slider_steps = slider_steps

        if hasattr(self, "iteration_slider"):
            self.iteration_slider.start = self.iter_step
            self.iteration_slider.end = self.max_iter_steps
            self.iteration_slider.step = self.iter_step
            self.iteration_slider.value = self.max_iter_steps

    @staticmethod
    def __text2layers(layers_str):
        return tuple([int(i) for i in layers_str.split(",")])

    @staticmethod
    def __label2activation_str(label):
        """transoform string from button to string that classifier expects"""
        if label == ButtonStr.IDENTITY:
            return "identity"
        elif label == ButtonStr.SIGMOID:
            return "logistic"
        elif label == ButtonStr.TANH:
            return "tanh"
        else:
            return "relu"

    @staticmethod
    def __label2solver_str(label):
        if label == ButtonStr.LBFGS:
            return "lbfgs"
        elif label == ButtonStr.GRADIENT:
            return "sgd"
        else:
            return "adam"
