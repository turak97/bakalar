
from data_sandbox import ClassifierDataSandbox, RegressionDataSandbox
from basic_sublayouts import ClassifierSubLayout, RegressionSubLayout
from extended_regression_sublayouts import PolynomialRegression, SliderRegressionSubLayout
from extended_classifier_sublayouts import \
    NeuralClassifier, SvmClassifier, SliderClassifierSubLayout

from models import REG_MODELS, REG_SLIDERS, CLS_MODELS, CLS_SLIDERS

from copy import deepcopy

import sklearn


def classifier_data_sandbox(name, source_data, class_select_button):
    return ClassifierDataSandbox(name, source_data, class_select_button)


def regression_data_sandbox(name, source_data):
    return RegressionDataSandbox(name, source_data)


def cls_resolution(model_name, source_data):
    model = deepcopy(CLS_MODELS[model_name])
    """Special models."""
    if model_name == "Neural classification":
        return NeuralClassifier(model_name, model, source_data)
    elif model_name == "SVM classification":
        return SvmClassifier(model_name, model, source_data)

    """Generic model with sliders attached"""
    if model_name in CLS_SLIDERS:
        slider_params = CLS_SLIDERS[model_name]
        return SliderClassifierSubLayout(model_name, model, source_data, slider_params)

    """Total generic model"""
    return ClassifierSubLayout(model_name, model, source_data)


def reg_resolution(model_name, source_data):
    model = deepcopy(REG_MODELS[model_name])
    """Special models"""
    if model_name == "Polynomial regression":
        return PolynomialRegression(model_name, model, source_data)

    """Generic model with sliders attached"""
    if model_name in REG_SLIDERS:
        slider_params = REG_SLIDERS[model_name]
        return SliderRegressionSubLayout(model_name, model, source_data, slider_params)

    """Total generic model"""
    return RegressionSubLayout(model_name, model, source_data)
