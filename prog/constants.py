from bokeh import palettes

########## Those constants sets common behaviour (name of saved files, default button values...) ##########

SAVED_DATASETS_DIR_NAME = "my_datasets/"
SAVED_DATASET_FILE_NAME = "my_dataset.csv"

PALETTE = palettes.Category10[10]

"""used for app init in file bokeh_plot.py"""

INIT_DATASET_CLUSTERS_COUNT_DEF = 3

"""Data sandbox widget values"""

LASSO_SLIDER_START = 1
LASSO_SLIDER_END = 100
LASSO_SLIDER_STEP = 1  # positive integer expected
LASSO_SLIDER_STARTING_VAL = 10


CLUSTER_RANGE_X = (-200, 200)
CLUSTER_RANGE_Y = (-200, 200)
CLUSTER_RANGE_STEP = 1


FREEHAND_DENSITY_START = 1
FREEHAND_DENSITY_END = 100
FREEHAND_DENSITY_STEP = 1  # positive integer expected
FREEHAND_DENSITY_STARTING_VAL = 10

FREEHAND_DEVIATION_START = 0
FREEHAND_DEVIATION_END = 20
FREEHAND_DEVIATION_STEP = 1
FREEHAND_DEVIATION_STARTING_VAL = 2

ALPHA_DEF = 2
BETA_DEF = 2

"""Neural options"""

NEURAL_DEF_SLIDER_STEPS = 4
NEURAL_DEF_MAX_ITER_STEPS = 400
NEURAL_DEF_ACTIVATION = 2  # 0: identity, 1: sigmoid, 2: tanh, 3: linear
NEURAL_DEF_SOLVER = 2  # 0: lbfgs, 1: gradient descent, 2: adam
NEURAL_DEF_LAYERS = "10, 10"  # more layers: "20, 10, 5, 20"
LOSS_PRINT = "log"  # either "log", "app", None

"""Regression sublayouts"""

POLY_DEF_DGR = 3
POL_FROM_DGR = 1
POL_TO_DGR = 8

"""Classification sublayouts"""

KNN_DEF_NEIGHBOUR_N = 3


CLUSTER_SIZE_DEF = 20
CLUSTER_SIZE_MAX = 50
CLUSTER_DEV_DEF = 10  # cluster size plus minus
CLUSTER_DEV_MAX = int(CLUSTER_SIZE_MAX*0.66)

########## Those constants are for experiments (interesting app behaviour, but possible (bigger) instability) ########

"""used in generic_sublayouts.py"""

# Sets the detail of classifier image
MESH_STEP_SIZE = 0.1  # default: 0.05

# Sets the detail of regression line
LINE_POINTS = 1000  # default 1000

X_EXT = 0.5
Y_EXT = X_EXT

########## Those constants should NOT be changed ##########

MAX_CLUSTERS = 10

EMPTY_VALUE_COLOR = 'black'

# Used in DataSandbox and data_gen.py
UNIFORM_MODE = "Uniform"
BETA_MODE = "Beta"
