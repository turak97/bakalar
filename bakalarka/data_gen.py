import random
import math

import numpy as np


def classify(length, classes):
    return [str(random.randint(0, classes - 1)) for _ in range(length)]


# TODO: fce check values
# generuje clustry na x ose, ty potom mergne (a seradi), dogeneruje y hodnoty a k nim pricte dispersi
def polynom_data(polynom=np.array([1/100, 1/5, -1, 1]),
                 interval=(-400, 400),
                 clusters=3,
                 noise=0.2,
                 density=8,
                 density_vol=-1,  # density volatility = 2 ... density can be from 6 to 10
                 rand_seed=1):
    if density_vol < 0:
        density_vol = density//2

    x_min, x_max = interval
    scale = x_max - x_min

    x_arr = np.array([])
    clusters_made = 0
    while clusters_made < clusters:
        x_from, x_to = gen_interval(interval, scale / clust_range_coef(clusters))
        cluster = gen_clust_beta(x_from, x_to, density, density_vol)
        x_arr = np.append(x_arr, cluster)
        clusters_made += 1

    # sort array (merge clusters) and cut overlapping clusters
    x_arr = np.sort(x_arr)

    f = np.poly1d(polynom)
    y_arr = f(x_arr)

    if noise != 0:
        y_arr = make_noise(y_arr, noise)

    return [x_arr, y_arr]


def polynom_line(model_coef, x_from, x_to, steps=1000):
    model_coef = model_coef[::-1]  # numpy poly1d expects coefficients in reversed order
    f = np.poly1d(model_coef)
    x_plot = np.linspace(x_from, x_to, steps)
    y_plot = f(x_plot)
    return x_plot, y_plot


def cluster_data(x_interval=(-100, 100),
                 y_interval=(-100, 100),
                 clusters=3,
                 av_cluster_size=15,
                 clust_size_vol=-1,
                 size=0.2,
                 rand_seed=1):
    if clust_size_vol < 0:
        clust_size_vol = av_cluster_size//2

    x_min, x_max = x_interval
    y_min, y_max = y_interval
    scale = x_max - x_min

    x_values = np.array([])
    y_values = np.array([])
    classification = []
    # classification = np.empty(dtype=np.int, shape=0)

    clusters_made = 0
    while clusters_made < clusters:
        x_from, x_to = gen_interval(x_interval, scale / clust_range_coef(clusters))
        y_from, y_to = gen_interval(y_interval, scale / clust_range_coef(clusters))

        cluster_size = av_cluster_size + abs(random.randint(-clust_size_vol, clust_size_vol))
        x_cluster = gen_clust_beta(x_from, x_to, cluster_size)
        y_cluster = gen_clust_beta(y_from, y_to, cluster_size)

        x_values = np.append(x_values, x_cluster)
        y_values = np.append(y_values, y_cluster)
        classification += [str(clusters_made)] * cluster_size
        # classification = np.append(classification, [np.int(clusters_made)] * cluster_size)

        clusters_made += 1

    return [x_values, y_values, classification]


# TODO pridat vyjimky do intervalu (leva mez vetsi nez prava)
# generate coefficients of the polynom
def gen_coefficients(degr, ferocity):
    coef = []
    for i in range(0, degr + 1):  # don't forget the constant
        coef.append(random.uniform(ferocity/1000, ferocity*2))

    return np.array(coef)


def make_noise(y_values, coef):
    max_noise = (max(y_values) - min(y_values))*coef
    for i in range(len(y_values)):
        y_values[i] += random.uniform(-max_noise, max_noise)
    return y_values


# expects array with values from interval [0, 1]
# reshapes them to the given interval
def rerange(arr, interval):
    coef = interval[1] - interval[0]
    shift = interval[0]
    for i in range(len(arr)):
        arr[i] = coef * arr[i] + shift
    return arr


# interval = interval, where sub-interval will be generated
# scale = scale of interval +- scale/2
def gen_interval(interval, scale):
    x_min, x_max = interval
    center = random.uniform(x_min, x_max)
    x_from, x_to = (random.betavariate(2, 2), random.betavariate(2, 2))
    if x_from > x_to:
        x_from, x_to = x_to, x_from

    x_from, x_to = rerange([x_from, x_to], (center - scale/2, center + scale/2))
    return x_from, x_to


def gen_clust_beta(from_, to_, size):
    vals = np.zeros(shape=size)
    alpha = random.randint(1, 10)
    beta = random.randint(1, 10)
    for i in range(len(vals)):
        vals[i] = random.betavariate(alpha, beta)
    vals = rerange(vals, (from_, to_))
    return vals


# def cut_out_values(array, min_, max_):
#     min_i = np.argmax(array >= min_)
#     max_i = np.argmax(array > max_)
#     # dealing with situation when max_ is bigger than max(arr) because np.argmax returns 0 then
#     # when min_ is bigger than max(arr) likewise
#     if min_i == 0 and array[-1] <= min_:
#         min_i = len(array)
#     if max_i == 0 and array[0] <= max_:
#         max_i = len(array)
#     return array[min_i:max_i]


# helps clusters to be smaller when there is more of them
def clust_range_coef(num_of_clusters):
    if num_of_clusters <= 1:
        return 1
    return math.log(num_of_clusters, 2)


# expects val_from <= val_to and bound from <= bound_to
# returns True, if there is empty intersection of intervals
def out_of_bounds(val_from, val_to, bound_from, bound_to):
    return val_to < bound_from or val_from > bound_to


def insert_point_x_sorted(x_arr, y_arr, x_val, y_val):
    i = np.searchsorted(x_arr, x_val)
    x_arr = np.insert(x_arr, i, x_val)
    y_arr = np.insert(y_arr, i, y_val)
    return x_arr, y_arr
