
import numpy as np

from bokeh.layouts import row, column
from bokeh.models import PointDrawTool, Button, LassoSelectTool, Div, \
    CheckboxButtonGroup
from bokeh.plotting import figure

from constants import MESH_STEP_SIZE, EMPTY_VALUE_COLOR, X_EXT, Y_EXT

# import matplotlib.pyplot as plt


# TODO: togglebutton misto radiobuttongroup


class SubLayout:
    """Abstract class for classifier sub layouts, regressive sub layouts and data sandbox"""
    def __init__(self, name, source_data):
        self.name = name
        self.source_data = source_data
        self._x_ext = X_EXT
        self._y_ext = Y_EXT

        self.layout = self._layout_init()

    def _layout_init(self):
        fig_layout = self._init_figure()
        fit_layout = self._init_fit_layout()
        button_layout = self._init_button_layout()
        return column(fig_layout, fit_layout, button_layout)

    def update_renderer_colors(self):
        pass

    def refit(self):
        pass

    def immediate_update(self):
        return 0 in self._immediate_update.active  # immediate update option is at index 0

    def _init_fit_layout(self):
        self._fit_button = Button(label="Solo Fit", button_type="success")
        self._fit_button.on_click(self.refit)

        self._immediate_update = CheckboxButtonGroup(
            labels=["Immediate update when new points added"], active=[])

        return row(self._immediate_update, self._fit_button)

    def _init_button_layout(self):
        return row()

    def _init_figure(self):
        self._fig = figure(tools="pan,wheel_zoom,save,reset,box_zoom")
        self._lasso = LassoSelectTool()
        self._fig.add_tools(self._lasso)
        return self._fig

    def _info(self, message):
        print(self.name + " " + self._fig.id + ": " + message)


class ClassifierSubLayout(SubLayout):
    class ImageData:
        """Class for image representation of model results"""
        def __init__(self, x_min, x_max, y_min, y_max):
            self.x_min = x_min
            self.x_max = x_max
            self.y_min = y_min
            self.y_max = y_max
            self.dw = x_max - x_min
            self.dh = y_max - y_min

            self.xx, self.yy = np.meshgrid(np.arange(x_min, x_max, MESH_STEP_SIZE),
                                           np.arange(y_min, y_max, MESH_STEP_SIZE))

            self.d = []

        def add_d(self, d):
            self.d.append(d)

    def __init__(self, name, classifier, source_data):
        """Create attribute self._classifier and self.layout
        plus self.name, self.source_data and self.fig from super

        data and source_data references are necessary to store due to updating
        figure based on user input (e.g. different neural activation function)
        """
        SubLayout.__init__(self, name, source_data)

        # add original data to the figure and prepare PointDrawTool to make them interactive
        # this renderer MUST be the FIRST one
        move_circle = self._fig.circle('x', 'y', color='color', source=source_data.plot_source, size=7)
        point_draw_tool = PointDrawTool(renderers=[move_circle], empty_value=EMPTY_VALUE_COLOR, add=True)
        self._fig.add_tools(point_draw_tool)

        self._info("Initialising sublayout and fitting data...")

        self._classifier = classifier
        self.refit()
        self._info("Initialising DONE")

    def refit(self):
        self._info("Updating model and fitting data...")
        self._update_classifier_params()
        self._figure_update()
        self._info("Fitting and updating DONE")

    def _update_classifier_params(self):
        pass

    def update_renderer_colors(self):
        """
        triggers update of data_source for the figure color update
        """
        for renderer, new_d in zip(self._fig.renderers[1:], self._img_data.d):
            renderer.data_source.data['image'] = [new_d]

    def _figure_update(self):
        """Figure must have an 'image' renderer as SECOND (at index 1) renderer,
        where will be directly changed data
        """
        (min_x, max_x), (min_y, max_y) = self.source_data.get_min_max_x(), self.source_data.get_min_max_y()
        self._img_data = self.ImageData(min_x, max_x,
                                        min_y, max_y)

        self._fit_and_render(1)  # renderer at index 1 contains classifier image

    def _fit_and_render(self, renderer_i):
        """Fits the model, render image and add/update image to the figure
        expects attribute self.__img_data
        """
        self._info("Fitting data and updating figure, step: " + str(renderer_i))

        cls_X, classification = self.source_data.data_to_classifier_fit()
        self._classifier.fit(cls_X, classification)

        raw_d = self._classifier.predict(np.c_[self._img_data.xx.ravel(),
                                               self._img_data.yy.ravel()])
        self._img_data.add_d(raw_d.reshape(self._img_data.xx.shape))

        if len(self._fig.renderers) - 1 < renderer_i:
            self._new_fig_renderer(renderer_i - 1)
        else:
            self._update_fig_renderer(renderer_i)

    def _new_fig_renderer(self, d_index):
        # create a new image renderer
        self._fig.image(image=[self._img_data.d[d_index]], x=self._img_data.x_min, y=self._img_data.y_min,
                        dw=self._img_data.dw, dh=self._img_data.dh,
                        color_mapper=self.source_data.color_mapper, global_alpha=0.5)

    def _update_fig_renderer(self, i):
        """Update image data by directly changing them in the figure renderers"""
        img_patch = {
            'image': [(0, self._img_data.d[i - 1])]
        }
        self._fig.renderers[i].data_source.patch(img_patch)

        self._fig.renderers[i].glyph.update(
            color_mapper=self.source_data.color_mapper,
            x=self._img_data.x_min,
            y=self._img_data.y_min,
            dw=self._img_data.dw,
            dh=self._img_data.dh
        )

    def _indie_plot(self):
        pass
        # TODO: method for independent check of image
        # plt.plot()
        # plt.contourf(xx, yy, d, cmap=plt.cm.coolwarm, alpha=0.8)
        # plt.scatter(data.x_data, data.y_data, c=classification, cmap=plt.cm.coolwarm)
        # plt.show()


class RegressionSubLayout(SubLayout):
    def __init__(self, name, model, source_data):

        SubLayout.__init__(self, name, source_data)

        # add original data to the figure and prepare PointDrawTool to make them interactive
        # this renderer MUST be the FIRST one
        move_circle = self._fig.circle('x', 'y', source=source_data.plot_source, size=7)
        point_draw_tool = PointDrawTool(renderers=[move_circle], empty_value=EMPTY_VALUE_COLOR, add=True)
        self._fig.add_tools(point_draw_tool)

        self._model = model
        self.refit()
        self._info("Initialising DONE")

    def _init_figure(self):
        x_min, x_max = self.source_data.get_min_max_x()
        x_range_extension = (x_max - x_min) * self._x_ext
        x_range = (x_min - x_range_extension, x_max + x_range_extension,)
        y_range = x_range  # figure scope should be square
        self._fig = figure(x_range=x_range, y_range=y_range)
        return self._fig

