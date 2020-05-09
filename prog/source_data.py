
from bokeh.models import CategoricalColorMapper, ColumnDataSource

import numpy as np

from in_n_out import save_source
import data_gen as dg


class SourceData:
    def __init__(self, df, column_names):
        self.x, self.y = column_names
        self.immediate_update = False

        self.plot_source = ColumnDataSource(
            data={
                self.x: df[self.x].tolist(),
                self.y: df[self.y].tolist()
            }
        )
        # self.plot_source.remove('index')

        self.plot_source_trigger = None

    def set_plot_source_trigger(self, f):
        self.plot_source.on_change('data', f)
        self.plot_source_trigger = f

    def get_min_max_x(self):
        return min(self.plot_source.data[self.x]), max(self.plot_source.data[self.x])

    def get_min_max_y(self):
        return min(self.plot_source.data[self.y]), max(self.plot_source.data[self.y])

    def save_source(self, file_name):
        abs_path = save_source(self.plot_source, file_name)
        return abs_path


class RegressionSourceData(SourceData):
    def __init__(self, df, column_names=('x', 'y')):
        SourceData.__init__(self, df, column_names)

    def data_to_regression_fit(self):
        x_np = np.asarray(self.plot_source.data[self.x])
        x_np = x_np[:, np.newaxis]
        y_np = np.asarray(self.plot_source.data[self.y])
        return x_np, y_np

    def replace_data(self, x, y):
        self.plot_source.remove_on_change('data', self.plot_source_trigger)
        self.plot_source.update(
            data=dict(
                x=x.tolist(),
                y=y.tolist()
            )
        )
        self.plot_source.on_change('data', self.plot_source_trigger)

        self.plot_source.stream({
            self.x: [],
            self.y: []
        })

    def append_data(self, x_new, y_new):
        new_data = {
            self.x: x_new.tolist(),
            self.y: y_new.tolist()
        }
        self.plot_source.stream(new_data)

    def append_polygon_data(self, vertices, density, distribution_params):
        x_new, y_new = dg.polygon_data(vertices, cluster_size=density, distribution_params=distribution_params)
        self.append_data(x_new, y_new)

    def append_freehand_data(self, x_line, y_line, size, volatility,
                             distribution_params):
        x_new, y_new = dg.line2cluster(x_line, y_line, size, volatility, distribution_params)
        self.append_data(x_new, y_new)


class ClassificationSourceData(SourceData):
    def __init__(self, df, palette, column_names=('x', 'y', 'classification')):
        SourceData.__init__(self, df, column_names=column_names[:2])
        self.palette = palette
        self.classification = column_names[2]

        uniq_values = sorted(list(set(df[self.classification])))
        self.color_mapper = CategoricalColorMapper(palette=self.palette, factors=uniq_values)
        self.color_dict = self.__uniq_vals2color_dict(uniq_values)

        self.plot_source.add(df[self.classification].tolist(), self.classification)
        self.plot_source.add([self.color_dict[val] for val in df[self.classification]], 'color')

    def data_to_classifier_fit(self):
        data = self.plot_source.data
        cls_X = np.array([[data[self.x][i], data[self.y][i]] for i in range(len(data[self.x]))])
        return cls_X, self.plot_source.data[self.classification]

    def new_clusters(self, count, density, volatility, x_range, y_range, replace=False):
        x, y, classification = dg.cluster_data(x_interval=x_range, y_interval=y_range,
                                               clusters=count,
                                               av_cluster_size=density,
                                               clust_size_vol=volatility)

        if replace:
            self.replace_data(x, y, classification)
        else:
            self.append_data(x, y, classification)

    def append_polygon_data(self, vertices, cls, density, distribution_params):
        x_new, y_new = dg.polygon_data(vertices, cluster_size=density, distribution_params=distribution_params)
        classification_new = [cls] * len(x_new)
        self.append_data(x_new, y_new, classification_new)

    def append_freehand_data(self, x_line, y_line, cls, size, volatility, distribution_params):
        x_new, y_new = dg.line2cluster(x_line, y_line, size, volatility, distribution_params)
        classification_new = [cls] * len(x_new)

        self.append_data(x_new, y_new, classification_new)

    def append_data(self, x_new, y_new, classification_new):
        colors = [self.color_dict[cls] for cls in classification_new]
        new_data = {
            self.x: x_new.tolist(),
            self.y: y_new.tolist(),
            self.classification: classification_new,
            'color': colors
        }
        self.plot_source.stream(new_data)

    def replace_data(self, x, y, classification):
        uniq_values = self.uniq_values()
        for cls in classification:
            if cls not in uniq_values:
                uniq_values.append(cls)
        self.color_mapper = CategoricalColorMapper(palette=self.palette, factors=uniq_values)
        self.color_dict = self.__uniq_vals2color_dict(uniq_values)

        self.plot_source.remove_on_change('data', self.plot_source_trigger)
        self.plot_source.update(
            data=dict(
                x=x.tolist(),
                y=y.tolist(),
                classification=classification,
                color=[self.color_dict[val] for val in classification]
            )
        )
        self.plot_source.on_change('data', self.plot_source_trigger)

        self.append_data(np.empty(shape=0), np.empty(shape=0), [])

    def uniq_values(self):
        return sorted(self.color_dict.keys())

    def add_new_class(self, class_name):
        """Add a new color option."""
        prev_last_index = len(self.uniq_values()) - 1
        self.color_dict[class_name] = self.palette[prev_last_index + 1]
        self.color_mapper = CategoricalColorMapper(palette=self.palette, factors=self.uniq_values())

    def update_cls_newly_added(self, new_class, new_i):
        """Update color in newly added data,
        new_i is starting index of the new rows (... attribute 'color' is EMPTY_VALUE_COLOR).
        """
        self.plot_source.remove_on_change('data', self.plot_source_trigger)
        source_len = len(self.plot_source.data[self.x])

        patches = {
            self.classification: [(i, new_class) for i in range(new_i, source_len)],
            'color': [(i, self.color_dict[new_class]) for i in range(new_i, source_len)]
        }
        self.plot_source.patch(patches)

        self.plot_source.on_change('data', self.plot_source_trigger)  # subscribe again

    def replace_color(self, old_color, new_color):
        """Replace old color with new color in source"""
        self.plot_source.remove_on_change('data', self.plot_source_trigger)
        patch = []
        for i, color in zip(range(len(self.plot_source.data['color'])), self.plot_source.data['color']):
            if color == old_color:  # find old color occurrences and add then into the patch
                patch.append((i, new_color))

        patches = {
            'color': patch
        }
        self.plot_source.patch(patches)

        for cls, color in self.color_dict.items():
            if color == old_color:
                self.color_dict[cls] = new_color

        tmp_palette = list(self.palette)
        for i in range(len(tmp_palette)):
            if tmp_palette[i] == old_color:
                tmp_palette[i] = new_color
        self.palette = tuple(tmp_palette)

        self.color_mapper.palette = self.palette

        self.plot_source.on_change('data', self.plot_source_trigger)

    def __uniq_vals2color_dict(self, uniq_values):
        """Map classification values to integers such as
        ["setosa", "virginica", "setosa"] -> {0: 'setosa', 1: 'virginica'}
        """
        values_dict = {}
        for uq, i in zip(uniq_values, range(len(uniq_values))):
            values_dict[uq] = self.palette[i]
        return values_dict
