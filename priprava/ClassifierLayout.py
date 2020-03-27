import numpy as np
from bokeh.layouts import row
from bokeh.models import PointDrawTool, LinearColorMapper
from bokeh.plotting import figure

import matplotlib.pyplot as plt

import Layout


MESH_STEP_SIZE = 0.05  # detail of plotted picture


def concat(x, y):
    return np.array([[x[i], y[i]] for i in range(len(x))])

#
# def classifier_layout(name, data, plot_info, classifier):
#     fig = figure()
#
#     fig_update(
#         fig=fig, data=data,
#         classifier=classifier,
#         mesh_step_size=MESH_STEP_SIZE,
#         palette=plot_info.palette
#     )
#
#     # add original data to the figure and prepare PointDrawTool to make them interactive
#     move_circle = fig.circle('x', 'y', color='color', source=plot_info.plot_source, size=7)
#     point_draw_tool = PointDrawTool(renderers=[move_circle], empty_value='added', add=True)
#     fig.add_tools(point_draw_tool)
#
#     cls_layout = row(fig)
#
#     cls_lay = ClassifierLayout(name=name, classifier=classifier, layout=cls_layout, fig=fig)
#
#     return cls_lay


class ClassifierLayout(Layout.SubLayout):
    def __init__(self, name, classifier, data, plot_info):
        self.name = name
        self.classifier = classifier
        self.fig = figure()

        self.data_update(
            data=data,
            plot_info=plot_info
        )

        # add original data to the figure and prepare PointDrawTool to make them interactive
        move_circle = self.fig.circle('x', 'y', color='color', source=plot_info.plot_source, size=7)
        point_draw_tool = PointDrawTool(renderers=[move_circle], empty_value='added', add=True)
        self.fig.add_tools(point_draw_tool)

        self.layout = row(self.fig)

    def data_update(self, data, plot_info):
        """
        figure must have an 'image' renderer as ONLY or FIRST renderer, where will be directly changed data
        or figure must have no renderers
        """
        X = concat(data.x_data, data.y_data)

        classification = data.classification
        x_min, x_max = data.x_data.min() - 1, data.x_data.max() + 1
        y_min, y_max = data.y_data.min() - 1, data.y_data.max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, MESH_STEP_SIZE),
                             np.arange(y_min, y_max, MESH_STEP_SIZE))
        classifier = self.classifier.fit(X, classification)
        d = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        d = d.reshape(xx.shape)

        # TODO: method for independent check of image
        # plt.plot()
        # plt.contourf(xx, yy, d, cmap=plt.cm.coolwarm, alpha=0.8)
        # plt.scatter(data.x_data, data.y_data, c=classification, cmap=plt.cm.coolwarm)
        # plt.show()

        if not self.fig.renderers:
            # create a new image renderer
            mapper = LinearColorMapper(palette=plot_info.palette, low=0, high=3)
            self.fig.image(image=[d], x=x_min, y=y_min, dw=x_max - x_min, dh=y_max - y_min,
                           color_mapper=mapper, global_alpha=0.5)
        else:
            # updating image data by directly changing them in figure
            # TODO: make a function for this
            fig.renderers[0].data_source.data['image'] = [d]
            fig.renderers[0].glyph.x = x_min
            fig.renderers[0].glyph.y = y_min
            fig.renderers[0].glyph.dw = x_max - x_min
            fig.renderers[0].glyph.dh = y_max - y_min

