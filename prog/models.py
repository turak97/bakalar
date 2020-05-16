
from sklearn.svm import SVC, SVR
from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

import math


CLS_MODELS = {
    "SVM classification": SVC(),
    "K nearest neighbours": KNeighborsClassifier(),
    "Stochastic gradient descent": SGDClassifier(),
    "Naive Bayes (Gaussian)": GaussianNB(),
    "Neural classification": MLPClassifier(warm_start=True, tol=0, n_iter_no_change=math.inf)
}

# ("name of the figure", (start, end, step, value))
CLS_SLIDERS = {
    "K nearest neighbours": ("n_neighbors", (1, 3, 1, 3))
}

REG_MODELS = {
    "SVM regression": SVR(),
    "K nearest neighbours": KNeighborsRegressor(),
    "Polynomial regression": Pipeline([('poly', PolynomialFeatures(degree=1)),
                                       ('linear', LinearRegression(fit_intercept=False))]),
    "Neural regression": MLPRegressor(warm_start=True, n_iter_no_change=math.inf)
}

REG_SLIDERS = {
    "K nearest neighbours": ("n_neighbors", (1, 10, 1, 3))
}
