
from extended_classifier_sublayouts import \
    NeuralClassifier, SvmClassifier, BayesClassifier, KnnClassifier
from data_sandbox import DataSandbox
from basic_sublayouts import ClassifierSubLayout

import polynomial_regression as pr


def data_sandbox(name, plot_info, class_select_button):
    return DataSandbox(name, plot_info, class_select_button)


def resolution(model, name, plot_info):
    if not isinstance(model, str):
        return ClassifierSubLayout(
            name=name, classifier=model, plot_info=plot_info
        )
    type_, kind = model.split(".")
    if type_ == "cls":
        if kind == "neural":
            return NeuralClassifier(
                name=name, plot_info=plot_info
            )
        elif kind == "svm":
            return SvmClassifier(
                name=name, plot_info=plot_info
            )
        elif kind == "knn":
            return KnnClassifier(
                name=name, plot_info=plot_info
            )
        elif kind == "bayes":
            return BayesClassifier(
                name=name, plot_info=plot_info
            )
        else:
            return None
    elif type_ == "reg":
        return pr.polynomial_layout(
            name=name,
            plot_info=plot_info
        )
    else:
        return None
