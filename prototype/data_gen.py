import random

from sklearn import datasets
from sklearn import svm
from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import math

import numpy as np

# TODO pridat vyjimky do intervalu (leva mez vetsi nez prava)


# generate coefficients of the polynom
def gen_coefficients(degr, ferocity):
    coef = []
    for i in range(0, degr + 1):  # don't forget the constant
        coef.append(random.uniform(ferocity/1000, ferocity*2))

    return np.array(coef)


def disperse(y_values, coef):
    max_disp = (max(y_values) - min(y_values))*coef
    for i in range(len(y_values)):
        y_values[i] += random.uniform(-max_disp, max_disp)
    return y_values


# expects array with values from interval [0, 1]
# reshapes them to the given interval
def rerange(arr, interval):
    coef = interval[1] - interval[0]
    shift = interval[0]
    for i in range(len(arr)):
        arr[i] = coef * arr[i] + shift
    return arr


# center = point around which shall be the interval generated
# scale = interval will be generated in center +- scale/2
def gen_interval(center, scale):
    x_min, x_max = (random.betavariate(2, 2), random.betavariate(2, 2))
    if x_min > x_max:
        x_min, x_max = x_max, x_min

    x_min, x_max = rerange([x_min, x_max], (center - scale/2, center + scale/2))
    return x_min, x_max


def gen_clust_beta(from_, to_, count):
    vals = np.zeros(shape=count)
    alpha = random.randint(1, 10)
    beta = random.randint(1, 10)
    for i in range(len(vals)):
        vals[i] = random.betavariate(alpha, beta)
    vals = rerange(vals, (from_, to_))
    return vals


def cut_out_values(array, min_, max_):
    min_i = np.argmax(array >= min_)
    max_i = np.argmax(array > max_)
    # dealing with situation when max_ is bigger than max(arr) because np.argmax returns 0 then
    # when min_ is bigger than max(arr) likewise
    if min_i == 0 and array[-1] <= min_:
        min_i = len(array)
    if max_i == 0 and array[0] <= max_:
        max_i = len(array)
    return array[min_i:max_i]


# helps clusters to be smaller when there is more of them
def clust_range_coef(num_of_clusters):
    if num_of_clusters <= 1:
        return 1
    return math.log(num_of_clusters, 2)


# expects val_from <= val_to and bound from <= bound_to
# returns True, if there is empty intersection of intervals
def out_of_bounds(val_from, val_to, bound_from, bound_to):
    return val_to < bound_from or val_from > bound_to


# generuje clustry na x ose, ty potom mergne (a seradi), dogeneruje y hodnoty a k nim pricte dispersi
def polynom_data(polynom=np.array([1/100, 1/5, -1, 1]),
                 interval=(-400, 400),
                 clusters=3,
                 disp=0.2,
                 density=10,
                 rand_seed=False,
                 num_of_outliers=2):
    if rand_seed:
        pass #TODO
    x_min, x_max = interval
    scale = x_max - x_min

    x_arr = np.array([])
    clusters_made = 0
    while clusters_made < clusters:
        x_from, x_to = gen_interval(random.uniform(x_min, x_max), scale / clust_range_coef(clusters))
        cluster = gen_clust_beta(x_from, x_to, density)
        cluster = cut_out_values(cluster, x_min, x_max)
        if len(cluster) == 0:
            # sometimes happens when only 1 cluster is generated
            # when generating more, the chances are slim
            continue
        x_arr = np.append(x_arr, cluster)
        clusters_made += 1

    # sort array (merge clusters) and cut overlapping clusters
    x_arr = np.sort(x_arr)

    f = np.poly1d(polynom)
    y_arr = f(x_arr)

    if disp != 0:
        y_arr = disperse(y_arr, disp)

    return [x_arr, y_arr]


def polynom_line(model_coef, x_from, x_to, steps=1000):
    model_coef = model_coef[::-1]  # numpy poly1d expects coefficients in reverse order
    f = np.poly1d(model_coef)
    x_plot = np.linspace(x_from, x_to, steps)
    y_plot = f(x_plot)
    return x_plot, y_plot


# fit model and return coefficients
def polynomial_model_coeff(degree, x_data, y_data):
    model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                      ('linear', LinearRegression(fit_intercept=False))])
    model = model.fit(x_data[:, np.newaxis], y_data)
    return model.named_steps['linear'].coef_


def insert_point_x_sorted(x_arr, y_arr, x_val, y_val):
    i = np.searchsorted(x_arr, x_val)
    x_arr = np.insert(x_arr, i, x_val)
    y_arr = np.insert(y_arr, i, y_val)
    return x_arr, y_arr
