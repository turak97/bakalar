
import numpy as np

from bokeh.layouts import row, column
from bokeh.models import PointDrawTool, Button, LassoSelectTool, Div, \
    CheckboxButtonGroup, Slider, TextInput, Toggle, RadioButtonGroup
from bokeh.plotting import figure

from math import ceil

from constants import MESH_STEP_SIZE, LINE_POINTS, EMPTY_VALUE_COLOR, X_EXT, Y_EXT, NEURAL_DEF_SOLVER, \
    NEURAL_DEF_ACTIVATION, NEURAL_DEF_LAYERS, NEURAL_DEF_MAX_ITER_STEPS, NEURAL_DEF_SLIDER_STEPS


# import matplotlib.pyplot as plt


class SubLayout:
    """Base class for ModelSubLayout and DataSandbox"""
    def __init__(self, name, source_data):
        self.name = name
        self.source_data = source_data

        self.layout = self._layout_init()

    def _layout_init(self):
        return row(self._init_figure())

    def _init_figure(self):
        self._fig = figure(match_aspect=True, tools="pan,wheel_zoom,save,reset,box_zoom")
        self._lasso = LassoSelectTool()
        self._fig.add_tools(self._lasso)
        return self._fig

    def _info(self, message):
        print(self.name + " " + self._fig.id + ": " + message)


class ModelSubLayout(SubLayout):
    """Base class for classifier sub layouts, regressive subl_ayouts"""
    def __init__(self, name, model, source_data):
        SubLayout.__init__(self, name, source_data)

        self._model = model

        self._x_ext = X_EXT
        self._y_ext = Y_EXT

    """Methods used by GeneralLayout"""

    def refit(self):
        self._info("Updating model and fitting data...")
        self._update_model_params()
        self._figure_update()
        self._info("Fitting and updating DONE")

    def immediate_update(self):
        """Returns whether figure should be immediately updated (after dataset change)"""
        return 0 in self._immediate_update.active  # immediate update option is at index 0

    """Methods for initialising layout"""

    def _layout_init(self):
        fig_layout = self._init_figure()
        fit_layout = self._init_fit_layout()
        button_layout = self._init_button_layout()
        return column(fig_layout, fit_layout, button_layout)

    def _update_model_params(self):
        pass

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


class SliderLike:
    def __init__(self, slider_params):
        self._model_attr, slider_attr = slider_params
        self._start, self._end, self._step, self._value = slider_attr

    def _init_button_layout(self):
        self._slider = Slider(
            title=self._model_attr,
            start=self._start, end=self._end, step=self._step, value=self._value
        )
        self._slider.on_change("value", self._slider_change)
        return self._slider

    def _slider_change(self, attr, old, new):
        visible = new
        self._set_visible_renderer(visible)

    def _set_visible_renderer(self, visible):
        pass


class NeuralLike:
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

    def __init__(self):
        """Creates attribute self.name, self.classifier, self.fig, self.layout self.source_data from super"""
        # initialise iteration parameters for slider and classifier fitting
        self._update_iteration_params(NEURAL_DEF_MAX_ITER_STEPS, NEURAL_DEF_SLIDER_STEPS)
        self._logarithmic_steps = False

    def _init_button_layout(self):
        """Creates buttons bellow the figure, sets the trigger functions on them
        and add them to the subLayout."""
        total_width = 500
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
        activation_text = Div(text="Activation function in hidden layers:")
        activation_group = column(activation_text, self._activation_button)

        self._solver_button = RadioButtonGroup(
            labels=[self.ButtonStr.LBFGS, self.ButtonStr.GRADIENT, self.ButtonStr.ADAM],
            active=NEURAL_DEF_SOLVER,
            width=total_width
        )
        solver_text = Div(text="Weigh optimization solver:")
        solver_group = column(solver_text, self._solver_button)

        return column(slider_group,
                      layers_input, activation_group,
                      solver_group)

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

    def _set_visible_renderer(self, visible):
        pass

    @staticmethod
    def _text2layers(layers_str):
        return tuple([int(i) for i in layers_str.split(",")])

    @staticmethod
    def _label2activation_str(label):
        """transform string from button to string that classifier expects"""
        if label == NeuralLike.ButtonStr.IDENTITY:
            return "identity"
        elif label == NeuralLike.ButtonStr.SIGMOID:
            return "logistic"
        elif label == NeuralLike.ButtonStr.TANH:
            return "tanh"
        else:
            return "relu"

    @staticmethod
    def _label2solver_str(label):
        if label == NeuralLike.ButtonStr.LBFGS:
            return "lbfgs"
        elif label == NeuralLike.ButtonStr.GRADIENT:
            return "sgd"
        else:
            return "adam"


class ClassifierSubLayout(ModelSubLayout):
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

    def __init__(self, name, model, source_data):
        """Create attribute self._model and self.layout
        plus self.name, self.source_data and self.fig from super

        data and source_data references are necessary to store due to updating
        figure based on user input (e.g. different neural activation function)
        """
        ModelSubLayout.__init__(self, name, model, source_data)

        # add original data to the figure and prepare PointDrawTool to make them interactive
        # this renderer MUST be the FIRST one
        move_circle = self._fig.circle(source_data.x, source_data.y, color='color', source=source_data.plot_source, size=7)
        point_draw_tool = PointDrawTool(renderers=[move_circle], empty_value=EMPTY_VALUE_COLOR, add=True)
        self._fig.add_tools(point_draw_tool)

        self.refit()
        self._info("Initialising DONE")

    """Methods used by ClassifierGeneralLayout"""

    def update_renderer_colors(self):
        """Triggers update of data_source for the figure color update.
        """
        for renderer, new_image in zip(self._fig.renderers[1:], self._img_data.images):
            renderer.data_source.data['image'] = [new_image]

    """Methods for updating figure"""

    def _figure_update(self):
        """Figure must have an 'image' renderer as SECOND (at index 1) renderer,
        where will be directly changed data
        """
        (min_x, max_x), (min_y, max_y) = self.source_data.get_min_max_x(), self.source_data.get_min_max_y()
        self._img_data = self.ImageData(min_x, max_x,
                                        min_y, max_y,
                                        self._x_ext, self._y_ext)

        self._fit_and_render(1)

    def _fit_and_render(self, renderer_i):
        """Fits the model, render image and add/update image to the figure
        expects attribute self.__img_data
        """
        self._info("Fitting data and updating figure, step: " + str(renderer_i))

        cls_X, classification = self.source_data.data_to_classifier_fit()
        self._model.fit(cls_X, classification)

        raw_image = self._model.predict(np.c_[self._img_data.xx.ravel(),
                                              self._img_data.yy.ravel()])
        self._img_data.add_image(raw_image.reshape(self._img_data.xx.shape))

        if len(self._fig.renderers) - 1 < renderer_i:
            self._new_fig_renderer(renderer_i - 1)  # images are indexed from 0, image renderers from 1
        else:
            self._update_fig_renderer(renderer_i)

    def _new_fig_renderer(self, d_index):
        # create a new image renderer
        self._fig.image(image=[self._img_data.images[d_index]], x=self._img_data.x_min, y=self._img_data.y_min,
                        dw=self._img_data.dw, dh=self._img_data.dh,
                        color_mapper=self.source_data.color_mapper, global_alpha=0.5)

    def _update_fig_renderer(self, i):
        """Update image data by directly changing them in the figure renderers"""
        img_patch = {
            'image': [(0, self._img_data.images[i - 1])]  # images are indexed from 0, image renderers from 1
        }
        self._fig.renderers[i].data_source.patch(img_patch)

        self._fig.renderers[i].glyph.update(
            color_mapper=self.source_data.color_mapper,
            x=self._img_data.x_min,
            y=self._img_data.y_min,
            dw=self._img_data.dw,
            dh=self._img_data.dh
        )


class RegressionSubLayout(ModelSubLayout):
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

    def __init__(self, name, model, source_data):

        ModelSubLayout.__init__(self, name, model, source_data)

        # add original data to the figure and prepare PointDrawTool to make them interactive
        # this renderer MUST be the FIRST one
        move_circle = self._fig.circle(source_data.x, source_data.y, source=source_data.plot_source, size=7)
        point_draw_tool = PointDrawTool(renderers=[move_circle], empty_value=EMPTY_VALUE_COLOR, add=True)
        self._fig.add_tools(point_draw_tool)

        self.refit()
        self._info("Initialising DONE")

    """Methods for updating figure"""

    def _figure_update(self):
        """Figure must have an 'image' renderer as SECOND (at index 1) renderer,
        where will be directly changed data..
        """
        (x_min, x_max) = self.source_data.get_min_max_x()
        self._line_data = self.LineData(x_min, x_max, self._x_ext)

        self._fit_and_render(1)

    def _fit_and_render(self, renderer_i):
        """Fits the model, render image and add/update image to the figure
        expects attribute self.__img_data
        """
        self._info("Fitting data and updating figure, step: " + str(renderer_i))

        x_data, y_data = self.source_data.data_to_regression_fit()
        self._model.fit(x_data, y_data)

        y_line = self._model.predict(np.c_[self._line_data.xx.ravel()])
        self._line_data.add_line(y_line)

        if len(self._fig.renderers) - 1 < renderer_i:
            self._new_fig_renderer(renderer_i - 1)
        else:
            self._update_fig_renderer(renderer_i)

    def _new_fig_renderer(self, line_i):
        """Creates a new line renderer from data at line_i in self._line_data.lines"""
        x_data, y_data = self._line_data.lines[line_i]
        self._fig.line(x_data, y_data)

    def _update_fig_renderer(self, i):
        """Update image data by directly changing them in the figure renderers"""
        x_data, y_data = self._line_data.lines[i - 1]
        self._fig.renderers[i].data_source.update(
            data=dict(
                x=x_data,
                y=y_data
            )
        )
