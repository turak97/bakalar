
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
    "Stochastic gradiend descent": SGDClassifier(),
    "K nearest neighbours": KNeighborsClassifier(n_neighbors=3),
    "Naive Bayes (Gaussian)": GaussianNB(),
    "Neural classification": MLPClassifier()
}

REG_MODELS = {
    "Polynomial regression": Pipeline([('poly', PolynomialFeatures(degree=1)),
                                       ('linear', LinearRegression(fit_intercept=False))]),
    "K nearest neighbours": KNeighborsRegressor()
}
