

from bokeh.layouts import row, column
import numpy as np
import polynomial_regression as pr
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from NeuralClassifierLayout import NeuralClassifierLayout
from ClassifierLayout import ClassifierLayout


class PlotInfo:
    def __init__(self, plot_source, pol_min_degree, pol_max_degree,
                 palette, x_extension, y_extension, mesh_step_size):
        self.plot_source = plot_source
        self.pol_min_degree = pol_min_degree
        self.pol_max_degree = pol_max_degree
        self.palette = palette
        self.x_extension = x_extension
        self.y_extension = y_extension
        self.mesh_step_size = mesh_step_size

    # new_i is starting index of the new rows (where attribute 'color' is 'added')
    def update_color(self, new_class, f, new_i):
        self.plot_source.remove_on_change('data', f)  # unsubscribe f, so there wont be an unwanted trigger

        source_len = len(self.plot_source.data['x'])

        patches = {
            'classification': [(i, new_class) for i in range(new_i, source_len)],
            'color': [(i, self.palette[new_class]) for i in range(new_i, source_len)]
        }
        self.plot_source.patch(patches)

        self.plot_source.on_change('data', f)  # subscribe again


def list_to_row(lay_list):
    return row([lay.layout for lay in lay_list])


def str2classifier(classifier_type):
    if classifier_type == "svc_linear":
        return svm.SVC(kernel='linear', C=1)
    elif classifier_type == "svc_poly":
        return svm.SVC(kernel='poly', C=1, gamma='auto')
    elif classifier_type == "svc_sigmoid":
        return svm.SVC(kernel='sigmoid', C=1, gamma='auto')
    elif classifier_type == "knn":
        return KNeighborsClassifier(3)
    elif classifier_type == "bayes":
        return GaussianNB()
    else:
        return None


def resolution(model, name, data, plot_info):
    if not isinstance(model, str):
        return model
    type_, kind = model.split(".")
    if type_ == "clas":
        if kind == "neural":
            return NeuralClassifierLayout(
                name=name,
                data=data, plot_info=plot_info,
            )
        else:
            return ClassifierLayout(name=name,
                                    classifier=str2classifier(kind),
                                    data=data, plot_info=plot_info)
    elif type_ == "reg":
        return pr.polynomial_layout(name=name,
                                    data=data, plot_info=plot_info)
    else:
        return None


def find_first(tuple_list, second):
    for t1, t2 in filter(lambda x: x is not None, tuple_list):
        if t2 == second:
            return t1
    return None
