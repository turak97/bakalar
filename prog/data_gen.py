# import random
import math

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np
import pandas as pd

from constants import BETA_MODE, UNIFORM_MODE


def line2cluster(x_line, y_line, size, deviation, distribution_params):
    line_len = len(x_line)
    mode, beta_rand, alpha, beta = distribution_params

    x_vals = np.zeros(shape=size)
    y_vals = np.zeros(shape=size)
    if beta_rand:
        alpha = np.random.randint(1, 10)
        beta = np.random.randint(1, 10)

    for i in range(size):
        line_i = 0
        if mode == BETA_MODE:
            rand_beta = np.random.beta(alpha, beta)  # nonuniform number from 0 to 1
            line_i = int(line_len * rand_beta)
        if mode == UNIFORM_MODE:
            line_i = np.random.randint(0, line_len)

        x_dev, y_dev = np.random.uniform(-deviation, deviation), np.random.uniform(-deviation, deviation)
        x_vals[i], y_vals[i] = x_line[line_i] + x_dev, y_line[line_i] + y_dev

    return x_vals, y_vals


# generates cluster inside a polygon
# generates only x and y values, NOT classification
def polygon_data(polygon_vertices, cluster_size=-1, distribution_params=(UNIFORM_MODE, True, 1, 1)):
    polygon = Polygon(polygon_vertices)
    mode, beta_rand, alpha, beta = distribution_params

    if cluster_size == 0:
        return [np.empty(dtype=int, shape=0), np.empty(dtype=int, shape=0)]

    if cluster_size < 0:
        cluster_size = max(int(polygon.area * np.random.uniform(0.3, 1.5)), 1)

    x_min, y_min, x_max, y_max = polygon.bounds

    x_values = np.zeros(shape=cluster_size)
    y_values = np.zeros(shape=cluster_size)
    if beta_rand:
        alpha = np.random.randint(1, 10)
        beta = np.random.randint(1, 10)

    points_created = 0
    while points_created < cluster_size:
        x, y = 0, 0
        if mode == UNIFORM_MODE:
            x, y = np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)
        if mode == BETA_MODE:
            x = x_min + (x_max - x_min) * np.random.beta(alpha, beta)
            y = y_min + (y_max - y_min) * np.random.beta(alpha, beta)
        point_candidate = Point(x, y)
        if not polygon.contains(point_candidate):
            continue
        x_values[points_created] = x
        y_values[points_created] = y
        points_created += 1

    return x_values, y_values


def cluster_data(x_interval=(-100, 100),
                 y_interval=(-100, 100),
                 clusters=3,
                 av_cluster_size=15,
                 clust_size_dev=-1,
                 classes=None):
    if clust_size_dev < 0:
        clust_size_dev = av_cluster_size//2

    if classes is None:
        classes = [str(i) for i in range(clusters)]

    x_min, x_max = x_interval
    scale = x_max - x_min

    x_values = np.array([])
    y_values = np.array([])
    classification = []

    clusters_made = 0
    while clusters_made < clusters:
        x_from, x_to = gen_interval(x_interval, scale / clust_range_coef(clusters))
        y_from, y_to = gen_interval(y_interval, scale / clust_range_coef(clusters))

        cluster_size = av_cluster_size + np.random.randint(-clust_size_dev, clust_size_dev + 1)
        x_cluster = gen_clust_beta(x_from, x_to, cluster_size)
        y_cluster = gen_clust_beta(y_from, y_to, cluster_size)

        x_values = np.append(x_values, x_cluster)
        y_values = np.append(y_values, y_cluster)
        classification += [classes[clusters_made]] * cluster_size

        clusters_made += 1

    return x_values, y_values, classification


def cluster_data_pandas(column_names,
                        x_interval=(-100, 100),
                        y_interval=(-100, 100),
                        clusters=3,
                        av_cluster_size=15,
                        clust_size_dev=-1):
    x_arr, y_arr, classification_arr = cluster_data(x_interval=x_interval,
                                                    y_interval=y_interval,
                                                    clusters=clusters,
                                                    av_cluster_size=av_cluster_size,
                                                    clust_size_dev=clust_size_dev)
    x_name, y_name, classification_name = column_names
    d = {x_name: x_arr, y_name: y_arr, classification_name: classification_arr}
    return pd.DataFrame(data=d)


def classify(length, classes_list):
    return [np.random.choice(classes_list) for _ in range(length)]


"""Helper functions"""


# TODO pridat vyjimky do intervalu (leva mez vetsi nez prava)
# generate coefficients of the polynom
def gen_coefficients(degr, ferocity):
    coef = []
    for i in range(0, degr + 1):  # don't forget the constant
        coef.append(np.random.uniform(ferocity/1000, ferocity*2))

    return np.array(coef)


def make_noise(y_values, coef):
    max_noise = (max(y_values) - min(y_values))*coef
    for i in range(len(y_values)):
        y_values[i] += np.random.uniform(-max_noise, max_noise)
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
    center = np.random.uniform(x_min, x_max)
    x_from, x_to = (np.random.beta(2, 2), np.random.beta(2, 2))
    if x_from > x_to:
        x_from, x_to = x_to, x_from

    x_from, x_to = rerange([x_from, x_to], (center - scale/2, center + scale/2))
    return x_from, x_to


def gen_clust_beta(from_, to_, size):
    vals = np.zeros(shape=size)
    alpha = np.random.randint(1, 10)
    beta = np.random.randint(1, 10)
    for i in range(len(vals)):
        vals[i] = np.random.beta(alpha, beta)
    vals = rerange(vals, (from_, to_))
    return vals


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


# # TODO: fce check values
# # generuje clustry na x ose, ty potom mergne (a seradi), dogeneruje y hodnoty a k nim pricte disperzi
# def polynom_data(polynom=np.array([1/100, 1/5, -1, 1]),
#                  interval=(-400, 400),
#                  clusters=3,
#                  noise=0.2,
#                  density=8,
#                  density_vol=-1,  # density volatility = 2 ... density can be from 6 to 10
#                  rand_seed=1):
#     if density_vol < 0:
#         density_vol = density//2
#
#     x_min, x_max = interval
#     scale = x_max - x_min
#
#     x_arr = np.array([])
#     clusters_made = 0
#     while clusters_made < clusters:
#         x_from, x_to = gen_interval(interval, scale / clust_range_coef(clusters))
#         cluster = gen_clust_beta(x_from, x_to, density)
#         x_arr = np.append(x_arr, cluster)
#         clusters_made += 1
#
#     # sort array (merge clusters) and cut overlapping clusters
#     x_arr = np.sort(x_arr)
#
#     f = np.poly1d(polynom)
#     y_arr = f(x_arr)
#
#     if noise != 0:
#         y_arr = make_noise(y_arr, noise)
#
#     return [x_arr, y_arr]

