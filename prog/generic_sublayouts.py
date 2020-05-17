
import numpy as np

from bokeh.layouts import row, column
from bokeh.models import PointDrawTool, Button, LassoSelectTool, Div, Select, \
    CheckboxButtonGroup, Slider, TextInput, Toggle, RadioButtonGroup
from bokeh.plotting import figure

from math import ceil

import copy

from constants import MESH_STEP_SIZE, LINE_POINTS, EMPTY_VALUE_COLOR, X_EXT, Y_EXT, NEURAL_DEF_SOLVER, \
    NEURAL_DEF_ACTIVATION, NEURAL_DEF_LAYERS, NEURAL_DEF_MAX_ITER_STEPS, NEURAL_DEF_SLIDER_STEPS, LOSS_PRINT, \
    POLY_DEF_DGR

from models import REG_MODELS, CLS_MODELS


# import matplotlib.pyplot as plt


class SubLayout:
    """Base class for ModelSubLayout and DataSandbox"""
    def __init__(self, subl_name):
        self.subl_name = subl_name

        self.layout = self._layout_init()

    def _layout_init(self):
        return row(self._init_figure())

    def _init_figure(self):
        self._fig = figure(match_aspect=True, tools="pan,wheel_zoom,save,reset,box_zoom")
        self._lasso = LassoSelectTool()
        self._fig.add_tools(self._lasso)
        return self._fig

    def _info(self, message):
        print(self.subl_name + " " + self._fig.id + ": " + message)


class ModelInterface:
    def __init__(self, model_name, source_data):
        self._model = None
        self._model_name = model_name
        self.source_data = source_data

        self._x_ext = X_EXT
        self._y_ext = Y_EXT

    def _init_circle_renderer(self, fig):
        pass

    def _init_model_params(self):
        """Initialise the model parameters before training."""
        pass

    def _init_data(self):
        pass

    def _init_model(self):
        pass

    def _fit(self):
        pass

    def _render(self, fig, renderer_i):
        pass

    def _set_visible_renderer(self, renderer_i):
        pass

    def _new_fig_renderer(self, fig, img_i):
        """Create a new image renderer.
        """
        pass

    def _update_fig_renderer(self, fig, i):
        """Update image data by directly changing them in the figure renderers.
        """
        pass


class RegressionLike(ModelInterface):
    class LineData:
        """Class for line representation of model results"""

        def __init__(self, x_min, x_max, x_ext):
            self.x_extension = (x_max - x_min) * x_ext
            self.x_min = x_min - self.x_extension
            self.x_max = x_max + self.x_extension
            self.xx = np.linspace(self.x_min, self.x_max, LINE_POINTS)
            self.lines = []

        def add_line(self, y_data):
            x_data, y_data = self.cut_y_extreme(self.xx, y_data)

            self.lines.append((x_data, y_data))

        def cut_y_extreme(self, x_data, y_data):
            """returns new numpy array without extreme values"""
            new_y_data = []
            new_x_data = []
            extreme_out = self.x_extension * 10
            for x, y in zip(x_data, y_data):
                if self.x_max + extreme_out > y > self.x_min - extreme_out:
                    new_x_data.append(x)
                    new_y_data.append(y)
            return np.asarray(new_x_data), np.asarray(new_y_data)

    def __init__(self, name, source_data):
        ModelInterface.__init__(self, name, source_data)

    def _init_circle_renderer(self, fig):
        """Add original data to the figure and prepare PointDrawTool to make them interactive
        this renderer MUST be the FIRST one.
        """
        move_circle = fig.circle(self.source_data.x, self.source_data.y,
                                 source=self.source_data.plot_source, size=7)
        point_draw_tool = PointDrawTool(renderers=[move_circle], empty_value=EMPTY_VALUE_COLOR, add=True)
        fig.add_tools(point_draw_tool)

    def _init_model(self):
        self._model = copy.deepcopy(REG_MODELS[self._model_name])

    def _init_data(self):
        (x_min, x_max) = self.source_data.get_min_max_x()
        self._line_data = self.LineData(x_min, x_max, self._x_ext)

    def _fit(self):
        x_data, y_data = self.source_data.data_to_regression_fit()
        self._model.fit(x_data, y_data)

    def _render(self, fig, renderer_i):
        y_line = self._model.predict(np.c_[self._line_data.xx.ravel()])
        self._line_data.add_line(y_line)

        if len(fig.renderers) - 1 < renderer_i:
            self._new_fig_renderer(fig, renderer_i - 1)
        else:
            self._update_fig_renderer(fig, renderer_i)

    def _new_fig_renderer(self, fig, line_i):
        """Creates a new line renderer from data at line_i in self._line_data.lines"""
        x_data, y_data = self._line_data.lines[line_i]
        fig.line(x_data, y_data)

    def _update_fig_renderer(self, fig, i):
        """Update image data by directly changing them in the figure renderers"""
        x_data, y_data = self._line_data.lines[i - 1]
        fig.renderers[i].data_source.update(
            data=dict(
                x=x_data,
                y=y_data
            )
        )


class ClassificationLike(ModelInterface):
    class ImageData:
        """Class for image representation of model results"""

        def __init__(self, x_min, x_max, y_min, y_max, x_ext, y_ext):
            x_extension = (x_max - x_min) * x_ext
            y_extension = (y_max - y_min) * y_ext
            self.x_min = x_min - x_extension
            self.x_max = x_max + x_extension
            self.y_min = y_min - y_extension
            self.y_max = y_max + y_extension

            self.dw = self.x_max - self.x_min
            self.dh = self.y_max - self.y_min
            self.xx, self.yy = np.meshgrid(np.arange(self.x_min, self.x_max, MESH_STEP_SIZE),
                                           np.arange(self.y_min, self.y_max, MESH_STEP_SIZE))

            self.images = []

        def add_image(self, d):
            self.images.append(d)

    def __init__(self, name, source_data):
        ModelInterface.__init__(self, name, source_data)

    def _init_circle_renderer(self, fig):
        """Add original data to the figure and prepare PointDrawTool to make them interactive
        this renderer MUST be the FIRST one.
        """
        move_circle = fig.circle(self.source_data.x, self.source_data.y,
                                 color='color', source=self.source_data.plot_source, size=7)
        point_draw_tool = PointDrawTool(renderers=[move_circle], empty_value=EMPTY_VALUE_COLOR, add=True)
        fig.add_tools(point_draw_tool)

    def _init_model(self):
        self._model = copy.deepcopy(CLS_MODELS[self._model_name])

    def _init_data(self):
        (min_x, max_x), (min_y, max_y) = self.source_data.get_min_max_x(), self.source_data.get_min_max_y()
        self._img_data = self.ImageData(min_x, max_x,
                                        min_y, max_y,
                                        self._x_ext, self._y_ext)

    def _fit(self):
        cls_X, classification = self.source_data.data_to_classifier_fit()
        self._model.fit(cls_X, classification)

    def _render(self, fig, renderer_i):
        raw_image = self._model.predict(np.c_[self._img_data.xx.ravel(),
                                              self._img_data.yy.ravel()])
        self._img_data.add_image(raw_image.reshape(self._img_data.xx.shape))

        if len(fig.renderers) - 1 < renderer_i:
            self._new_fig_renderer(fig, renderer_i - 1)
        else:
            self._update_fig_renderer(fig, renderer_i)

    def _new_fig_renderer(self, fig, img_i):
        """Create a new image renderer.
        """
        fig.image(image=[self._img_data.images[img_i]], x=self._img_data.x_min, y=self._img_data.y_min,
                  dw=self._img_data.dw, dh=self._img_data.dh,
                  color_mapper=self.source_data.color_mapper, global_alpha=0.5)

    def _update_fig_renderer(self, fig, i):
        """Update image data by directly changing them in the figure renderers.
        """
        img_patch = {
            'image': [(0, self._img_data.images[i - 1])]  # images are indexed from 0, image renderers from 1
        }
        fig.renderers[i].data_source.patch(img_patch)

        fig.renderers[i].glyph.update(
            color_mapper=self.source_data.color_mapper,
            x=self._img_data.x_min,
            y=self._img_data.y_min,
            dw=self._img_data.dw,
            dh=self._img_data.dh
        )


class ModelSubLayout(SubLayout):
    """Base class for classifier sub layouts, regressive subl_ayouts"""
    def __init__(self, model_name):
        SubLayout.__init__(self, model_name)

    """Methods used by GeneralLayout"""

    def refit(self):
        self._info("Training model and updating figure...")
        self._figure_update()
        self._info("Training and updating DONE")

    def immediate_update(self):
        """Returns whether figure should be immediately updated (after dataset change)"""
        return 0 in self._immediate_update.active  # immediate update option is at index 0

    """Methods for initialising layout"""

    def _layout_init(self):
        fig_layout = self._init_figure()
        fit_layout = self._init_fit_layout()
        button_layout = self._init_button_layout()
        return column(fig_layout, fit_layout, button_layout)

    def _init_button_layout(self):
        return row()

    def _init_fit_layout(self):
        self._fit_button = Button(label="Solo Fit", button_type="success")
        self._fit_button.on_click(self.refit)

        self._immediate_update = CheckboxButtonGroup(
            labels=["Immediate update on dataset change"], active=[])

        return row(self._immediate_update, self._fit_button)

    """Methods for updating figure"""

    def _figure_update(self):
        pass


class BasicSubLayout(ModelSubLayout, ModelInterface):
    def __init__(self, subl_name):
        ModelSubLayout.__init__(self, subl_name)
        self._init_circle_renderer(self._fig)

    def _figure_update(self):
        self._info("Initialising model and render data...")
        self._init_data()
        self._init_model()
        self._info("Updating model parameters...")
        self._init_model_params()
        self._info("Training model...")
        self._fit()
        self._info("Adding renderer...")
        self._render(self._fig, 1)


class SliderSubLayout(ModelSubLayout, ModelInterface):
    def __init__(self, subl_name, slider_params):
        self._model_attr, slider_attr = slider_params
        self._start, self._end, self._step, self._value = slider_attr

        ModelSubLayout.__init__(self, subl_name)
        self._init_circle_renderer(self._fig)

    def _init_button_layout(self):
        self._slider = Slider(
            title=self._model_attr,
            start=self._start, end=self._end, step=self._step, value=self._value
        )
        self._slider.on_change("value", self._slider_change)
        return self._slider

    def _figure_update(self):
        self._info("Initialising model and render data...")
        self._init_data()
        self._init_model()

        for value, i in zip(range(self._start, self._end + 1, self._step),
                            range(1, self._end + 1, self._step)):
            self._info("Setting model attribute: " + str(self._model_attr) + " to value " + str(value))
            setattr(self._model, self._model_attr, value)

            self._info("Training model...")
            self._fit()
            self._info("Adding renderer number " + str(i) + "...")
            self._render(self._fig, i)

        visible = int((self._slider.value - self._start)/self._step) + 1
        self._set_visible_renderer(visible)

    def _slider_change(self, attr, old, new):
        visible = int((new - self._start)/self._step) + 1
        self._set_visible_renderer(visible)

    def _set_visible_renderer(self, visible):
        for renderer, i in zip(self._fig.renderers[1:], range(1, len(self._fig.renderers))):
            if i == visible:
                renderer.visible = True
            else:
                renderer.visible = False


class NeuralSubLayout(ModelSubLayout, ModelInterface):
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

    def __init__(self, subl_name):
        """Creates attribute self.name, self.classifier, self.fig, self.layout self.source_data from super"""
        # initialise iteration parameters for slider and classifier fitting
        self._update_iteration_params(NEURAL_DEF_MAX_ITER_STEPS, NEURAL_DEF_SLIDER_STEPS)
        self._logarithmic_steps = False
        self._neural_data = []

        ModelSubLayout.__init__(self, subl_name)
        self._init_circle_renderer(self._fig)

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

        self._iteration_slider = Slider(start=self._iter_step, end=self._max_iter_steps,
                                        step=self._iter_step, value=self._max_iter_steps,
                                        title="Iterations", width=total_width)
        self._iteration_slider.on_change('value', self._slider_change)
        max_iteration_text = Div(text="Max iterations:")
        self._max_iterations_input = TextInput(value=str(self._max_iter_steps), width=63)
        number_step_text = Div(text="Slider steps:")
        self._slider_steps_input = TextInput(value=str(self._slider_steps), width=45)
        self._logarithm_button = Toggle(label="Logarithmic slider", width=55)
        slider_group = column(self._iteration_slider,
                              row(max_iteration_text, self._max_iterations_input,
                                  number_step_text, self._slider_steps_input,
                                  self._logarithm_button)
                              )

        self._layers_input = TextInput(value=NEURAL_DEF_LAYERS)
        layers_text = Div(text="Hidden layers sizes:")
        layers_input = column(layers_text, row(self._layers_input))

        self._activation_button = RadioButtonGroup(
            labels=[self.ButtonStr.IDENTITY, self.ButtonStr.SIGMOID,
                    self.ButtonStr.TANH, self.ButtonStr.LINEAR], active=NEURAL_DEF_ACTIVATION,
            width=total_width
        )
        activation_text = Div(text="Activation function:")
        activation_group = column(activation_text, self._activation_button)

        self._solver_button = RadioButtonGroup(
            labels=[self.ButtonStr.LBFGS, self.ButtonStr.GRADIENT, self.ButtonStr.ADAM],
            active=NEURAL_DEF_SOLVER,
            width=total_width
        )
        solver_text = Div(text="Weigh optimization solver:")
        solver_group = column(solver_text, self._solver_button)

        return column(loss_group,
                      slider_group,
                      layers_input, activation_group,
                      solver_group)

    def _figure_update(self):
        self._info("Initialising model and render data...")
        self._init_data()
        self._init_model()
        self._info("Updating model attributes...")
        self._init_model_params()

        self._logarithmic_steps = self._logarithm_button.active
        self._update_iteration_params(int(self._max_iterations_input.value),
                                      int(self._slider_steps_input.value))

        self._model.max_iter = self._iter_step
        prev = 0  # used only with logarithmic slider
        for iterations, renderer_i in zip(range(self._iter_step, self._max_iter_steps + 1,
                                                self._iter_step),
                                          range(1, self._slider_steps + 1)):  # first one is Circle
            if self._logarithmic_steps:
                """
                in fact it is not logarithmic (I chose this name because I find it rather intuitive).
                This option allows user to see the begging of the learning
                process when the changes are much more significant in more detail.
                For  5000 iterations max and 10 steps it will be:
                50, 111, 187, 285, 416, 600, 875, 1333, 2250, 5000
                """
                log_iter_total = int(iterations / (self._slider_steps - renderer_i + 1))
                self._model.max_iter = max(log_iter_total - prev,
                                           1)
                prev = log_iter_total

            if hasattr(self._model, "classes_"):
                del self._model.classes_  # necessary for MLPClassifier warm start

            self._info("Training model...")
            self._fit()
            self._info("Training DONE, total iterations: " + str(self._model.n_iter_))

            self._neural_data.append(self._model.loss_)

            self._info("Adding renderer number " + str(renderer_i) + "...")
            self._render(self._fig, renderer_i)
        self._set_visible_renderer(self._slider_steps)

    def _slider_change(self, attr, old, new):
        visible = int(self._iteration_slider.value / self._iter_step)
        self._set_visible_renderer(visible)

    def _update_iteration_params(self, max_iter_steps, slider_steps):
        """Update iteration parameters
        and if _iteration_slider is initialised (typically after first fit), update slider parameters.
        """

        self._iter_step = ceil(max_iter_steps/slider_steps)
        self._max_iter_steps = self._iter_step * slider_steps
        self._slider_steps = slider_steps
        try:
            self._iteration_slider.update(
                start=self._iter_step,
                end=self._max_iter_steps,
                step=self._iter_step,
                value=self._max_iter_steps
            )
        except AttributeError:
            pass

    def _init_model_params(self):
        new_activation = self._label2activation_str(
            self._activation_button.labels[self._activation_button.active]
        )
        self._model.activation = new_activation

        new_solver = self._label2solver_str(
            self._solver_button.labels[self._solver_button.active]
        )
        self._model.solver = new_solver

        self._model.hidden_layer_sizes = self._text2layers(self._layers_input.value)

    def _set_visible_renderer(self, visible):
        for renderer, i in zip(self._fig.renderers[1:], range(1, len(self._fig.renderers))):
            if i == visible:
                renderer.visible = True
                if LOSS_PRINT == 'app':
                    self._loss_text.update(text=str(self._neural_data[i - 1]))
                elif LOSS_PRINT == 'log':
                    self._info("Validation loss: " + str(self._neural_data[i - 1]))
            else:
                renderer.visible = False

        if self._logarithmic_steps:
            self._iteration_slider.show_value = False
            self._iteration_slider.title = "Iterations logarithmic: " + str(
                int(self._iteration_slider.value / (self._slider_steps - visible + 1)))
        else:
            self._iteration_slider.show_value = True
            self._iteration_slider.title = "Iterations"

    @staticmethod
    def _text2layers(layers_str):
        return tuple([int(i) for i in layers_str.split(",")])

    @staticmethod
    def _label2activation_str(label):
        """transform string from button to string that classifier expects"""
        if label == NeuralSubLayout.ButtonStr.IDENTITY:
            return "identity"
        elif label == NeuralSubLayout.ButtonStr.SIGMOID:
            return "logistic"
        elif label == NeuralSubLayout.ButtonStr.TANH:
            return "tanh"
        else:
            return "relu"

    @staticmethod
    def _label2solver_str(label):
        if label == NeuralSubLayout.ButtonStr.LBFGS:
            return "lbfgs"
        elif label == NeuralSubLayout.ButtonStr.GRADIENT:
            return "sgd"
        else:
            return "adam"


class SvmSubLayout(BasicSubLayout, ModelInterface):
    class ButtonStr:
        # kernel button
        LINEAR = "linear"
        POLY = "polynomial"
        RBF = "radial (rbf)"
        SIGMOID = "sigmoid"

    def __init__(self, model_name):
        BasicSubLayout.__init__(self, model_name)

    def _init_button_layout(self):
        """Creates buttons bellow the figure, sets the trigger functions on them
        and add them to the subLayout"""
        total_width = 500

        _kernel_text = Div(text="Algorithm kernel: ")
        self._kernel_button = RadioButtonGroup(
            labels=[self.ButtonStr.LINEAR, self.ButtonStr.POLY, self.ButtonStr.RBF, self.ButtonStr.SIGMOID],
            active=0, width=total_width
        )
        _kernel_group = column(_kernel_text, self._kernel_button)

        self._degree_button = Select(
            title="", value=str(POLY_DEF_DGR),
            options=[str(i) for i in range(20)], width=70)
        degree_text = Div(text="Degree (" + self.ButtonStr.POLY + " only): ")

        self._regularization_parameter_input = TextInput(value="1.0", width=75)
        regularization_parameter_text = Div(text="Regularization parameter (C): ")

        return column(
                      _kernel_group,
                      row(regularization_parameter_text,
                          self._regularization_parameter_input),
                      row(degree_text, self._degree_button)
                      )

    def _init_model_params(self):
        new_kernel = self._label2kernel_str(
            self._chosen_kernel()
        )
        self._model.kernel = new_kernel

        self._model.degree = int(self._degree_button.value)  # _degree_button has predefined values

        self._model.C = self._get_regularization()

    def _chosen_kernel(self):
        return self._kernel_button.labels[self._kernel_button.active]

    def _get_regularization(self):
        res = 1.0
        try:
            res = float(self._regularization_parameter_input.value)
        except ValueError:
            self._regularization_parameter_input.value = "1.0"
        return res

    @staticmethod
    def _label2kernel_str(label):
        if label == SvmSubLayout.ButtonStr.LINEAR:
            return "linear"
        elif label == SvmSubLayout.ButtonStr.POLY:
            return "poly"
        elif label == SvmSubLayout.ButtonStr.RBF:
            return "rbf"
        else:
            return "sigmoid"
