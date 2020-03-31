import numpy as np
from bokeh.models import PointDrawTool, LinearColorMapper, Button

# import matplotlib.pyplot as plt

from Layout import SubLayout


class ImageData:
    def __init__(self, x_min, x_max, y_min, y_max, mesh_step_size):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.dw = x_max - x_min
        self.dh = y_max - y_min

        self.xx, self.yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                                       np.arange(y_min, y_max, mesh_step_size))

        self.d = None

    def set_d(self, d):
        self.d = d


class ClassifierLayout(SubLayout):
    def __init__(self, name, classifier, data, plot_info):
        """
        creates attribute self.classifier and self.layout
        plus self.name, self.data, self.plot_info and self.fig from super

        data and plot_info references are necessary to store due to updating
        figure based on user input (e.g. different neural activation function)
        """
        SubLayout.__init__(self, name, data, plot_info)

        self._info("Initialising layout and fitting data...")
        self._init_button_layout()

        # add original data to the figure and prepare PointDrawTool to make them interactive
        # this renderer MUST be the FIRST one
        move_circle = self.fig.circle('x', 'y', color='color', source=plot_info.plot_source, size=7)
        point_draw_tool = PointDrawTool(renderers=[move_circle], empty_value='black', add=True)
        self.fig.add_tools(point_draw_tool)

        self.classifier = classifier
        self.figure_update()

    def figure_update(self):
        """
        figure must have an 'image' renderer as SECOND (at index 1) renderer,
        where will be directly changed data
        """
        self._info("Updating model and fitting data...")

        img_data = ImageData(self.data.x_data.min() - 1, self.data.x_data.max() + 1,
                             self.data.y_data.min() - 1, self.data.y_data.max() + 1,
                             self.plot_info.mesh_step_size)

        self._fit_and_render(img_data, 1)

        self._info("Done")

    def _fit_and_render(self, img_data, renderer_i):
        """fits the model, render image and add/update image to the figure"""
        self._info("Fitting data and updating figure, step: " + str(renderer_i))
        self.classifier.fit(self.data.cls_X, self.data.classification)

        raw_d = self.classifier.predict(np.c_[img_data.xx.ravel(),
                                              img_data.yy.ravel()])
        img_data.set_d(raw_d.reshape(img_data.xx.shape))
        if len(self.fig.renderers) - 1 < renderer_i:
            self._new_fig_renderer(img_data)
        else:
            self._update_fig_renderer(img_data, renderer_i)

    def _new_fig_renderer(self, img_data):
        # create a new image renderer
        self.fig.image(image=[img_data.d], x=img_data.x_min, y=img_data.y_min,
                       dw=img_data.dw, dh=img_data.dh,
                       color_mapper=self.plot_info.color_mapper, global_alpha=0.5)

    def _update_fig_renderer(self, img_data, i):
        # updating image data by directly changing them in figure
        self.fig.renderers[i].data_source.data['image'] = [img_data.d]
        self.fig.renderers[i].glyph.x = img_data.x_min
        self.fig.renderers[i].glyph.y = img_data.y_min
        self.fig.renderers[i].glyph.dw = img_data.dw
        self.fig.renderers[i].glyph.dh = img_data.dh

    def _refit(self):
        self.figure_update()

    def _init_button_layout(self):
        self.fit_button = Button(label="Fit", button_type="success")
        self.fit_button.on_click(self._refit)

        self.layout.children[2] = self.fit_button

    def _indie_plot(self):
        pass
        # TODO: method for independent check of image
        # plt.plot()
        # plt.contourf(xx, yy, d, cmap=plt.cm.coolwarm, alpha=0.8)
        # plt.scatter(data.x_data, data.y_data, c=classification, cmap=plt.cm.coolwarm)
        # plt.show()
