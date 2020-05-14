
from data_sandbox import ClassifierDataSandbox, RegressionDataSandbox

from regression_sublayouts import BasicRegression, SliderRegression, NeuralRegression, PolynomialRegression
from classifier_sublayouts import BasicClassification, SliderClassification, \
    SvmClassification, NeuralClassification

from models import REG_MODELS, REG_SLIDERS, CLS_MODELS, CLS_SLIDERS

import sklearn


def classifier_data_sandbox(name, source_data, class_select_button):
    return ClassifierDataSandbox(name, source_data, class_select_button)


def regression_data_sandbox(name, source_data):
    return RegressionDataSandbox(name, source_data)


def cls_resolution(model_name, source_data):
    """Special models."""
    if model_name == "Neural classification":
        return NeuralClassification(model_name, source_data)
    elif model_name == "SVM classification":
        return SvmClassification(model_name, source_data)

    """Generic model with sliders attached"""
    if model_name in CLS_SLIDERS:
        slider_params = CLS_SLIDERS[model_name]
        return SliderClassification(model_name, source_data, slider_params)

    """Total generic model"""
    return BasicClassification(model_name, source_data)


def reg_resolution(model_name, source_data):
    """Special models"""
    if model_name == "Polynomial regression":
        return PolynomialRegression(model_name, source_data)
    if model_name == "Neural regression":
        return NeuralRegression(model_name, source_data)

    """Generic model with sliders attached"""
    if model_name in REG_SLIDERS:
        slider_params = REG_SLIDERS[model_name]
        return SliderRegression(model_name, source_data, slider_params)

    """Total generic model"""
    return BasicRegression(model_name, source_data)
