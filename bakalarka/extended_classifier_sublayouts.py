
from bokeh.models import RadioButtonGroup, TextInput, Button, Div, Slider, Toggle, Select
from bokeh.layouts import row, column

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from math import ceil

from basic_sublayouts import ClassifierSubLayout
from constants import NEURAL_DEF_ACTIVATION, NEURAL_DEF_LAYERS, NEURAL_DEF_MAX_ITER_STEPS, NEURAL_DEF_SLIDER_STEPS

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


# TODO: define all default values as constants

# TODO: u SVM pridat pod Fit button info "kernely: "
# TODO: prindat vseobecne info

# TODO: u knn zvazit moznosti algoritmu (vysledek stejny, mozna ponechat jen auto)

# TODO: sigmoid, gradient descent standartni parametry

class BayesClassifier(ClassifierSubLayout):
    def __init__(self, name, plot_info):
        classifier = GaussianNB()

        ClassifierSubLayout.__init__(self, name, classifier, plot_info)


class KnnClassifier(ClassifierSubLayout):
    class ButtonStr:
        # algorithm button
        BALLTREE = "ball tree"
        KDTREE = "kd tree"
        BRUTE = "brute force"
        AUTO = "auto"

    def __init__(self, name, plot_info):
        classifier = KNeighborsClassifier(n_neighbors=3)

        ClassifierSubLayout.__init__(self, name, classifier, plot_info)

    def _init_button_layout(self):
        """Creates buttons bellow the figure, sets the trigger functions on them
        and add them to the subLayout"""
        total_width = 500
        #
        # fit_button = Button(label="Fit", button_type="success", width=500)
        # fit_button.on_click(self.refit)

        self.__algo_button = RadioButtonGroup(
            labels=[self.ButtonStr.BALLTREE, self.ButtonStr.KDTREE, self.ButtonStr.BRUTE, self.ButtonStr.AUTO],
            active=3, width=total_width
        )

        self.__n_neighbors_button = Select(
            title="", value="3",
            options=[str(i) for i in range(1, 20)], width=70)
        n_neighbors_text = Div(text="Number of neighbors to use: ")

        return column(
                      self.__algo_button,
                      row(n_neighbors_text,
                          self.__n_neighbors_button)
                      )

    def _update_classifier_params(self):
        new_algo = self.__label2algo_str(
            self.__chosen_algo()
        )
        self._classifier.algorithm = new_algo

        self._classifier.n_neighbors = int(self.__n_neighbors_button.value)

    def __chosen_algo(self):
        return self.__algo_button.labels[self.__algo_button.active]

    @staticmethod
    def __label2algo_str(label):
        if label == KnnClassifier.ButtonStr.BALLTREE:
            return "ball_tree"
        elif label == KnnClassifier.ButtonStr.KDTREE:
            return "kd_tree"
        elif label == KnnClassifier.ButtonStr.BRUTE:
            return "brute"
        else:
            return "auto"


class SvmClassifier(ClassifierSubLayout):
    class ButtonStr:
        # kernel button
        LINEAR = "linear"
        POLY = "polynomial"
        RBF = "radial"
        SIGMOID = "sigmoid"

    def __init__(self, name, plot_info):
        classifier = SVC(kernel='linear')

        ClassifierSubLayout.__init__(self, name, classifier, plot_info)

    # def refit(self):
    #     self._info("Updating model and fitting data...")
    #     self.__update_classifier_params()
    #     self._figure_update()
    #     self._info("Fitting and updating DONE")

    def _init_button_layout(self):
        """Creates buttons bellow the figure, sets the trigger functions on them
        and add them to the subLayout"""
        total_width = 500
        #
        # fit_button = Button(label="Fit", button_type="success", width=500)
        # fit_button.on_click(self.refit)

        self.__kernel_button = RadioButtonGroup(
            labels=[self.ButtonStr.LINEAR, self.ButtonStr.POLY, self.ButtonStr.RBF, self.ButtonStr.SIGMOID],
            active=0, width=total_width
        )
        self.__degree_button = Select(
            title="", value="3",
            options=[str(i) for i in range(20)], width=70)
        degree_text = Div(text="Degree (" + self.ButtonStr.POLY + "): ")

        self.__regularization_parameter_input = TextInput(value="1.0", width=75)
        regularization_parameter_text = Div(text="Regularization parameter: ")

        return column(
                      self.__kernel_button,
                      row(regularization_parameter_text,
                          self.__regularization_parameter_input),
                      row(degree_text, self.__degree_button)
                      )

    def _update_classifier_params(self):
        new_kernel = self.__label2kernel_str(
            self.__chosen_kernel()
        )
        self._classifier.kernel = new_kernel

        self._classifier.degree = int(self.__degree_button.value)  # __degree_button has predefined values

        self._classifier.C = self.__get_regularization()

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
        if label == SvmClassifier.ButtonStr.LINEAR:
            return "linear"
        elif label == SvmClassifier.ButtonStr.POLY:
            return "poly"
        elif label == SvmClassifier.ButtonStr.RBF:
            return "rbf"
        else:
            return "sigmoid"


class NeuralClassifier(ClassifierSubLayout):
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

    def __init__(self, name, plot_info):
        """Creates attribute self.name, self.classifier, self.fig, self.layout self.plot_info from super"""
        classifier = MLPClassifier(random_state=1, tol=0)  # other parameters are gained from buttons
        # initialise iteration parameters for slider and classifier fitting
        self.__update_iteration_params(NEURAL_DEF_MAX_ITER_STEPS, NEURAL_DEF_SLIDER_STEPS)
        self.__logarithmic_steps = False

        ClassifierSubLayout.__init__(self, name, classifier, plot_info)
        self.__set_visible_renderer(self.__slider_steps)

    def _figure_update(self):
        self._info("Updating model and fitting data...")

        data = self.plot_info.plot_source.data
        self._img_data = self.ImageData(min(data['x']) - 1, max(data['x']) + 1,
                                        min(data['y']) - 1, max(data['y']) + 1)
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
                self._classifier.max_iter = int(iterations / (self.__slider_steps - renderer_i + 1))
            else:
                self._classifier.max_iter = iterations
            self._fit_and_render(renderer_i)

        self._info("Done")

    def refit(self):
        """Update iteration (max, slider step, ...) and classifier parameters then update figure"""
        self._info("Updating model and fitting data...")
        self.__logarithmic_steps = self.__logarithm_button.active

        self.__update_iteration_params(int(self.__max_iterations_input.value),
                                       int(self.__slider_steps_input.value))
        self._update_classifier_params()

        self._figure_update()
        self.__set_visible_renderer(self.__slider_steps)

        self._info("Fitting and updating DONE")

    def _init_button_layout(self):
        """Creates buttons bellow the figure, sets the trigger functions on them
        and add them to the subLayout"""
        total_width = 500
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
            active=2,
            width=total_width
        )
        solver_text = Div(text="Weigh optimization solver:")
        solver_group = column(solver_text, self.__solver_button)
        #
        # self.__fit_button = Button(label="Fit", button_type="success")
        # self.__fit_button.on_click(self.refit)

        return column(slider_group,
                      layers_input, activation_group,
                      solver_group)

    def __set_visible_renderer(self, visible):
        for renderer, i in zip(self._fig.renderers[1:], range(1, len(self._fig.renderers))):
            if i == visible:
                renderer.visible = True
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
        and if __iteration_slider is initialised (typically after first fit), update slider parameters
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

    def _update_classifier_params(self):
        new_activation = self.__label2activation_str(
            self.__activation_button.labels[self.__activation_button.active]
        )
        self._classifier.activation = new_activation

        new_solver = self.__label2solver_str(
            self.__solver_button.labels[self.__solver_button.active]
        )
        self._classifier.solver = new_solver

        self._classifier.hidden_layer_sizes = self.__text2layers(self.__layers_input.value)

    @staticmethod
    def __text2layers(layers_str):
        return tuple([int(i) for i in layers_str.split(",")])

    @staticmethod
    def __label2activation_str(label):
        """transform string from button to string that classifier expects"""
        if label == NeuralClassifier.ButtonStr.IDENTITY:
            return "identity"
        elif label == NeuralClassifier.ButtonStr.SIGMOID:
            return "logistic"
        elif label == NeuralClassifier.ButtonStr.TANH:
            return "tanh"
        else:
            return "relu"

    @staticmethod
    def __label2solver_str(label):
        if label == NeuralClassifier.ButtonStr.LBFGS:
            return "lbfgs"
        elif label == NeuralClassifier.ButtonStr.GRADIENT:
            return "sgd"
        else:
            return "adam"
