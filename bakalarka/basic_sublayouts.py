
import numpy as np

from bokeh.layouts import row, column
from bokeh.models import PointDrawTool, Button, LassoSelectTool, Div
from bokeh.plotting import figure

from constants import MESH_STEP_SIZE, EMPTY_VALUE_COLOR

# import matplotlib.pyplot as plt


class SubLayout:
    """Abstract class for classifier sub layouts, regressive sub layouts and data sandbox"""
    def __init__(self, name, plot_info):
        self.name = name
        self.plot_info = plot_info
        self._fig = figure(tools="pan,wheel_zoom,save,reset,box_zoom")
        self._lasso = LassoSelectTool()
        self._fig.add_tools(self._lasso)
        self.layout = self._layout_init()
        self._init_button_layout()

    def _layout_init(self):
        # last one row() is for children needs changed in _init_button_layout
        return column(self._fig, row())

    def update_renderer_colors(self):
        pass

    def refit(self):
        pass

    def _init_button_layout(self):
        pass

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

    def __init__(self, name, classifier, plot_info):
        """Create attribute self._classifier and self.layout
        plus self.name, self.plot_info and self.fig from super

        data and plot_info references are necessary to store due to updating
        figure based on user input (e.g. different neural activation function)
        """
        SubLayout.__init__(self, name, plot_info)

        self._info("Initialising sublayout and fitting data...")

        # add original data to the figure and prepare PointDrawTool to make them interactive
        # this renderer MUST be the FIRST one
        move_circle = self._fig.circle('x', 'y', color='color', source=plot_info.plot_source, size=7)
        point_draw_tool = PointDrawTool(renderers=[move_circle], empty_value=EMPTY_VALUE_COLOR, add=True)
        self._fig.add_tools(point_draw_tool)

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
        data = self.plot_info.plot_source.data
        self._img_data = self.ImageData(min(data['x']) - 1, max(data['x']) + 1,
                                        min(data['y']) - 1, max(data['y']) + 1)

        self._fit_and_render(1)  # renderer at index 1 contains classifier image

    def _fit_and_render(self, renderer_i):
        """Fits the model, render image and add/update image to the figure
        expects attribute self.__img_data
        """
        self._info("Fitting data and updating figure, step: " + str(renderer_i))

        data = self.plot_info.plot_source.data
        cls_X = np.array([[data['x'][i], data['y'][i]] for i in range(len(data['x']))])

        self._classifier.fit(cls_X, self.plot_info.plot_source.data['classification'])

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
                        color_mapper=self.plot_info.color_mapper, global_alpha=0.5)

    def _update_fig_renderer(self, i):
        """Update image data by directly changing them in the figure renderers"""
        img_patch = {
            'image': [(0, self._img_data.d[i - 1])]
        }
        self._fig.renderers[i].data_source.patch(img_patch)

        self._fig.renderers[i].glyph.update(
            color_mapper=self.plot_info.color_mapper,
            x=self._img_data.x_min,
            y=self._img_data.y_min,
            dw=self._img_data.dw,
            dh=self._img_data.dh
        )

    def _init_button_layout(self):
        self.fit_button = Button(label="Fit", button_type="success")
        self.fit_button.on_click(self.refit)

        self.layout.children[1] = self.fit_button

    def _indie_plot(self):
        pass
        # TODO: method for independent check of image
        # plt.plot()
        # plt.contourf(xx, yy, d, cmap=plt.cm.coolwarm, alpha=0.8)
        # plt.scatter(data.x_data, data.y_data, c=classification, cmap=plt.cm.coolwarm)
        # plt.show()
