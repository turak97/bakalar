
from extended_classifier_sublayouts import \
    NeuralClassifier, SvmClassifier, BayesClassifier, KnnClassifier, StochasticGDClassifier
from data_sandbox import ClassifierDataSandbox, RegressionDataSandbox
from basic_sublayouts import ClassifierSubLayout, RegressionSubLayout
from extended_regression_sublayouts import PolynomialRegression, KnnRegression

from models import REG_MODELS, CLS_MODELS

from copy import deepcopy

import sklearn


def classifier_data_sandbox(name, source_data, class_select_button):
    return ClassifierDataSandbox(name, source_data, class_select_button)


def regression_data_sandbox(name, source_data):
    return RegressionDataSandbox(name, source_data)


def cls_resolution(model_name, source_data):
    model = deepcopy(CLS_MODELS[model_name])
    if model_name == "Neural classification":
        return NeuralClassifier(model_name, model, source_data)
    elif model_name == "SVM classification":
        return SvmClassifier(model_name, model, source_data)
    elif model_name == "K nearest neighbours":
        return KnnClassifier(model_name, model, source_data)
    elif model_name == "Naive Bayes (Gaussian)":
        return BayesClassifier(model_name, model, source_data)
    elif model_name == "Stochastic gradiend descent":
        return StochasticGDClassifier(model_name, model, source_data)
    else:
        return None


def reg_resolution(model_name, source_data):
    model = deepcopy(REG_MODELS[model_name])
    if model_name == "Polynomial regression":
        return PolynomialRegression(model_name, model, source_data)
    elif model_name == "K nearest neighbours":
        return KnnRegression(model_name, model, source_data)
    else:
        return RegressionSubLayout(model_name, model, source_data)


# def str2model_class(model_name):
#     module_name, model_name = model_name.rsplit('.')
#     module = getattr(sklearn, module_name)
#     model_class = getattr(module, model_name)
#     return model_class
