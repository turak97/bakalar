
from extended_classifier_sublayouts import \
    NeuralClassifier, SvmClassifier, BayesClassifier, KnnClassifier
from data_sandbox import ClassifierDataSandbox
from basic_sublayouts import ClassifierSubLayout
from extended_regression_sublayouts import PolynomialRegression

import polynomial_regression as pr


def classifier_data_sandbox(name, source_data, class_select_button):
    return ClassifierDataSandbox(name, source_data, class_select_button)


def regression_data_sandbox(name, source_data):
    return DataSandbox(name, source_data)


def resolution(model, name, source_data):
    if not isinstance(model, str):
        return ClassifierSubLayout(
            name=name, classifier=model, source_data=source_data
        )
    type_, kind = model.split(".")
    if type_ == "cls":
        if kind == "neural":
            return NeuralClassifier(
                name=name, source_data=source_data
            )
        elif kind == "svm":
            return SvmClassifier(
                name=name, source_data=source_data
            )
        elif kind == "knn":
            return KnnClassifier(
                name=name, source_data=source_data
            )
        elif kind == "bayes":
            return BayesClassifier(
                name=name, source_data=source_data
            )
        else:
            return None
    elif type_ == "reg":
        return PolynomialRegression(
            name=name,
            source_data=source_data
        )
    else:
        return None
