
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


CLS_MODELS = {
    "SVM classification": SVC(),
    "Stochastic gradient descent": SGDClassifier(),
    "K nearest neighbours": KNeighborsClassifier(),
    "Naive Bayes (Gaussian)": GaussianNB(),
    "Neural classification": MLPClassifier()
}

# ("name of the figure", (start, end, step, value))
CLS_SLIDERS = {
    "K nearest neighbours": ("n_neighbors", (1, 3, 1, 3))
}

REG_MODELS = {
    "Polynomial regression": Pipeline([('poly', PolynomialFeatures(degree=1)),
                                       ('linear', LinearRegression(fit_intercept=False))]),
    "K nearest neighbours": KNeighborsRegressor()
}

