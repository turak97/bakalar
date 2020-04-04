
from bokeh.layouts import row, column
from bokeh.models import CategoricalColorMapper, ColumnDataSource
from sklearn import svm

import StandartClassifierSubLayouts
from ClassifierLayout import ClassifierSubLayout

import polynomial_regression as pr


class PlotInfo:
    def __init__(self, plot_source_init_data,
                 uniq_values, pol_min_degree, pol_max_degree,
                 palette, x_extension, y_extension, mesh_step_size):
        self.pol_min_degree = pol_min_degree
        self.pol_max_degree = pol_max_degree
        self.palette = palette
        self.x_extension = x_extension
        self.y_extension = y_extension
        self.mesh_step_size = mesh_step_size
        self.immediate_update = False
        self.uniq_values = uniq_values  # TODO: Del?
        self.color_mapper = CategoricalColorMapper(palette=self.palette, factors=uniq_values)

        self.color_dict = self.__uniq_vals2color_dict()
        self.plot_source = ColumnDataSource(
            data=dict(
                x=plot_source_init_data[0],
                y=plot_source_init_data[1],
                classification=plot_source_init_data[2],  # .toList()
                color=[self.color_dict[val] for val in plot_source_init_data[2]]
            )
        )

    def add_new_color(self, class_name):
        """add a new color possibility"""
        self.uniq_values.append(class_name)
        self.color_mapper = CategoricalColorMapper(palette=self.palette, factors=self.uniq_values)
        self.color_dict[class_name] = self.palette[len(self.uniq_values) - 1]

    def update_color_newly_added(self, new_class, new_i, f):
        """updates color in newly added data
        new_i is starting index of the new rows (where attribute 'color' is 'added')
        """
        self.plot_source.remove_on_change('data', f)  # unsubscribe f, so there wont be an unwanted trigger

        source_len = len(self.plot_source.data['x'])

        patches = {
            'classification': [(i, new_class) for i in range(new_i, source_len)],
            'color': [(i, self.color_dict[new_class]) for i in range(new_i, source_len)]
        }
        self.plot_source.patch(patches)

        self.plot_source.on_change('data', f)  # subscribe again

    def replace_color(self, old_color, new_color, f):
        self.plot_source.remove_on_change('data', f)
        patch = []
        for i, color in zip(range(len(self.plot_source.data['color'])), self.plot_source.data['color']):
            if color == old_color:
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

        self.plot_source.on_change('data', f)

    def __uniq_vals2color_dict(self):
        """Map classification values to integers such as
        ["setosa", "virginica", "setosa"] -> {0: 'setosa', 1: 'virginica'}
        """
        values_dict = {}
        for uq, i in zip(self.uniq_values, range(len(self.uniq_values))):
            values_dict[uq] = self.palette[i]
        return values_dict


def list_to_row(lay_list):
    return row([lay.layout for lay in lay_list])


def resolution(model, name, data, plot_info):
    if not isinstance(model, str):
        return ClassifierSubLayout(
            name=name, classifier=model, data=data, plot_info=plot_info
        )
    type_, kind = model.split(".")
    if type_ == "cls":
        if kind == "neural":
            return StandartClassifierSubLayouts.NeuralClassifier(
                name=name, data=data, plot_info=plot_info,
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
