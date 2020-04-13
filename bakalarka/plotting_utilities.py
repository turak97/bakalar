
from bokeh.layouts import row, column
from bokeh.models import CategoricalColorMapper, ColumnDataSource

import StandartClassifierSubLayouts
from DataSandbox import DataSandbox
from ClassifierSubLayout import ClassifierSubLayout

from numpy import empty

import polynomial_regression as pr

# TODO: pekneji reprezentovat plot_source_init_data


class PlotInfo:
    def __init__(self, df,
                 pol_min_degree, pol_max_degree,
                 palette, x_extension, y_extension):
        self.pol_min_degree = pol_min_degree
        self.pol_max_degree = pol_max_degree
        self.palette = palette
        self.x_extension = x_extension
        self.y_extension = y_extension
        self.immediate_update = False

        uniq_values = sorted(list(set(df['classification'])))
        self.color_mapper = CategoricalColorMapper(palette=self.palette, factors=uniq_values)
        self.color_dict = self.__uniq_vals2color_dict(uniq_values)

        self.plot_source = ColumnDataSource(df)
        self.plot_source.remove('index')
        self.plot_source.add([self.color_dict[val] for val in df['classification']], 'color')

        self.plot_source_trigger = None

    def set_plot_source_trigger(self, f):
        self.plot_source.on_change('data', f)
        self.plot_source_trigger = f

    def replace_data(self, x, y, classification):
        uniq_values = sorted(list(set(classification)))
        self.color_mapper = CategoricalColorMapper(palette=self.palette, factors=uniq_values)
        self.color_dict = self.__uniq_vals2color_dict(uniq_values)

        self.plot_source.remove_on_change('data', self.plot_source_trigger)
        self.plot_source.update(
            data=dict(
                x=x,
                y=y,
                classification=classification,
                color=[self.color_dict[val] for val in classification]
            )
        )
        self.plot_source.on_change('data', self.plot_source_trigger)

        self.append_data(empty(shape=0), empty(shape=0), [])

    def append_data(self, x_new, y_new, classification_new):

        colors = [self.color_dict[cls] for cls in classification_new]
        new_data = {
            'x': x_new.tolist(),
            'y': y_new.tolist(),
            'classification': classification_new,
            'color': colors
        }

        self.plot_source.stream(new_data)

    def uniq_values(self):
        return sorted(self.color_dict.keys())

    def add_new_color(self, class_name):
        """add a new color possibility"""
        prev_last_index = len(self.uniq_values()) - 1
        self.color_dict[class_name] = self.palette[prev_last_index + 1]
        self.color_mapper = CategoricalColorMapper(palette=self.palette, factors=self.uniq_values())

    def update_color_newly_added(self, new_class, new_i):
        """updates color in newly added data
        new_i is starting index of the new rows (where attribute 'color' is 'added')
        """
        self.plot_source.remove_on_change('data', self.plot_source_trigger)  # unsubscribe f, so there wont be an unwanted trigger

        source_len = len(self.plot_source.data['x'])

        patches = {
            'classification': [(i, new_class) for i in range(new_i, source_len)],
            'color': [(i, self.color_dict[new_class]) for i in range(new_i, source_len)]
        }
        self.plot_source.patch(patches)

        self.plot_source.on_change('data', self.plot_source_trigger)  # subscribe again

    def replace_color(self, old_color, new_color):
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


def list_to_row(lay_list):
    return row([lay.layout for lay in lay_list])


def data_sandbox(name, data, plot_info, class_select_button):
    return DataSandbox(name, data, plot_info, class_select_button)


def resolution(model, name, data, plot_info):
    if not isinstance(model, str):
        return ClassifierSubLayout(
            name=name, classifier=model, data=data, plot_info=plot_info
        )
    type_, kind = model.split(".")
    if type_ == "cls":
        if kind == "neural":
            return StandartClassifierSubLayouts.NeuralClassifier(
                name=name, data=data, plot_info=plot_info
            )
        elif kind == "svm":
            return StandartClassifierSubLayouts.SvmClassifier(
                name=name, data=data, plot_info=plot_info
            )
        elif kind == "knn":
            return StandartClassifierSubLayouts.KnnClassifier(
                name=name, data=data, plot_info=plot_info
            )
        elif kind == "bayes":
            return StandartClassifierSubLayouts.BayesClassifier(
                name=name, data=data, plot_info=plot_info
            )
        else:
            return None
    elif type_ == "reg":
        return pr.polynomial_layout(
            name=name,
            data=data, plot_info=plot_info
        )
    else:
        return None


def find_first(tuple_list, second):
    for t1, t2 in filter(lambda x: x is not None, tuple_list):
        if t2 == second:
            return t1
    return None
