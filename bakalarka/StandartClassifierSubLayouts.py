
from bokeh.models import RadioButtonGroup, TextInput, Button, Div, Slider, Toggle, Select
from bokeh.layouts import row, column

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from math import ceil

from ClassifierSubLayout import ClassifierSubLayout, ImageData
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

    # def refit(self):
    #     self._info("Updating model and fitting data...")
    #     self.__update_classifier_params()
    #     self._figure_update()
    #     self._info("Fitting and updating DONE")

    def _init_button_layout(self):
        """creates buttons bellow the figure and sets the trigger functions on them"""
        total_width = 500

        fit_button = Button(label="Fit", button_type="success", width=500)
        fit_button.on_click(self.refit)

        self.algo_button = RadioButtonGroup(
            labels=[self.ButtonStr.BALLTREE, self.ButtonStr.KDTREE, self.ButtonStr.BRUTE, self.ButtonStr.AUTO],
            active=3, width=total_width
        )

        self.n_neighbors_button = Select(
            title="", value="3",
            options=[str(i) for i in range(1, 20)], width=70)
        n_neighbors_text = Div(text="Number of neighbors to use: ")

        self.layout.children[1] = column(fit_button,
                                         self.algo_button,
                                         row(n_neighbors_text,
                                             self.n_neighbors_button)
                                         )

    def _update_classifier_params(self):
        new_algo = self.__label2algo_str(
            self.__chosen_algo()
        )
        self.classifier.algorithm = new_algo

        self.classifier.n_neighbors = int(self.n_neighbors_button.value)

    def __chosen_algo(self):
        return self.algo_button.labels[self.algo_button.active]

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
        """creates buttons bellow the figure and sets the trigger functions on them"""
        total_width = 500

        fit_button = Button(label="Fit", button_type="success", width=500)
        fit_button.on_click(self.refit)

        self.kernel_button = RadioButtonGroup(
            labels=[self.ButtonStr.LINEAR, self.ButtonStr.POLY, self.ButtonStr.RBF, self.ButtonStr.SIGMOID],
            active=0, width=total_width
        )
        self.degree_button = Select(
            title="", value="3",
            options=[str(i) for i in range(20)], width=70)
        degree_text = Div(text="Degree (" + self.ButtonStr.POLY + "): ")

        self.__regularization_parameter_input = TextInput(value="1.0", width=75)
        regularization_parameter_text = Div(text="Regularization parameter: ")

        self.layout.children[1] = column(fit_button,
                                         self.kernel_button,
                                         row(regularization_parameter_text,
                                             self.__regularization_parameter_input),
                                         row(degree_text, self.degree_button)
                                         )

    def _update_classifier_params(self):
        new_kernel = self.__label2kernel_str(
            self.__chosen_kernel()
        )
        self.classifier.kernel = new_kernel

        self.classifier.degree = int(self.degree_button.value)  # degree_button has predefined values

        self.classifier.C = self.__get_regularization()

    def __chosen_kernel(self):
        return self.kernel_button.labels[self.kernel_button.active]

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
        """
        creates attribute self.name, self.classifier, self.fig, self.layout
        self.plot_info from super
        """
        classifier = MLPClassifier(random_state=1, tol=0)  # other parameters are gained from buttons
        # initialise iteration parameters for slider and classifier fitting
        self.__update_iteration_params(NEURAL_DEF_MAX_ITER_STEPS, NEURAL_DEF_SLIDER_STEPS)
        self.__logarithmic_steps = False

        ClassifierSubLayout.__init__(self, name, classifier, plot_info)
        self.__set_visible_renderer(self.slider_steps)

    def _figure_update(self):
        self._info("Updating model and fitting data...")

        data = self.plot_info.plot_source.data
        self._img_data = ImageData(min(data['x']) - 1, max(data['x']) + 1,
                                   min(data['y']) - 1, max(data['y']) + 1)
        for iterations, renderer_i in zip(range(self.iter_step, self.max_iter_steps + 1,
                                                self.iter_step),
                                          range(1, self.slider_steps + 1)):  # first one is Circle
            if self.__logarithmic_steps:
                """
                in fact it is not logarithmic (I chose this name because I find it rather intuitive).
                This option allows user to see the begging of the learning
                process when the changes are much more significant in more detail.
                For  5000 iterations max and 10 steps it will be:
                50, 111, 187, 285, 416, 600, 875, 1333, 2250, 5000
                """
                self.classifier.max_iter = int(iterations / (self.slider_steps - renderer_i + 1))
            else:
                self.classifier.max_iter = iterations
            self._fit_and_render(renderer_i)

        self._info("Done")

    def refit(self):
        self._info("Updating model and fitting data...")
        self.fit_button.disabled = True  # disabling button so there are peaceful conditions for fitting model
        self.__logarithmic_steps = self.logarithm_button.active

        self.__update_iteration_params(int(self.max_iterations_input.value),
                                       int(self.slider_steps_input.value))
        self._update_classifier_params()

        self._figure_update()
        self.__set_visible_renderer(self.slider_steps)

        self.fit_button.disabled = False
        self._info("Fitting and updating DONE")

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

        self.layers_input = TextInput(value=NEURAL_DEF_LAYERS)
        layers_text = Div(text="Hidden layers sizes:")
        layers_input = column(layers_text, row(self.layers_input))

        self.activation_button = RadioButtonGroup(
            labels=[self.ButtonStr.IDENTITY, self.ButtonStr.SIGMOID,
                    self.ButtonStr.TANH, self.ButtonStr.LINEAR], active=NEURAL_DEF_ACTIVATION,
            width=total_width
        )
        activation_text = Div(text="Activation function in hidden layers:")
        activation_group = column(activation_text, self.activation_button)

        self.solver_button = RadioButtonGroup(
            labels=[self.ButtonStr.LBFGS, self.ButtonStr.GRADIENT, self.ButtonStr.ADAM],
            active=2,
            width=total_width
        )
        solver_text = Div(text="Weigh optimization solver:")
        solver_group = column(solver_text, self.solver_button)

        self.fit_button = Button(label="Fit", button_type="success")
        self.fit_button.on_click(self.refit)

        # add all of those at the position in the sublayout
        self.layout.children[1] = column(self.fit_button, slider_group,
                                         layers_input, activation_group,
                                         solver_group)

    def __set_visible_renderer(self, visible):
        for renderer, i in zip(self.fig.renderers[1:], range(1, len(self.fig.renderers))):
            if i == visible:
                renderer.visible = True
            else:
                renderer.visible = False

        if self.__logarithmic_steps:
            self.iteration_slider.show_value = False
            self.iteration_slider.title = "Iterations logarithmic: " + str(
                int(self.iteration_slider.value / (self.slider_steps - visible + 1)))
        else:
            self.iteration_slider.show_value = True
            self.iteration_slider.title = "Iterations"

    def __slider_change(self, attr, old, new):
        visible = int(self.iteration_slider.value / self.iter_step)
        self.__set_visible_renderer(visible)

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

    def _update_classifier_params(self):
        new_activation = self.__label2activation_str(
            self.activation_button.labels[self.activation_button.active]
        )
        self.classifier.activation = new_activation

        new_solver = self.__label2solver_str(
            self.solver_button.labels[self.solver_button.active]
        )
        self.classifier.solver = new_solver

        self.classifier.hidden_layer_sizes = self.__text2layers(self.layers_input.value)

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
