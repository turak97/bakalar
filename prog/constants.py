from bokeh import palettes

# TODO: nejake konstanty dat zpatky do soubory, kam patri (kde to ma smysl)

########## Those constants sets common behaviour (name of saved files, ...) ##########

SAVED_DATASETS_DIR_NAME = "def_datasets"
SAVED_DATASET_FILE_NAME = "my_dataset.csv"

PALETTE = palettes.Category10[10]

########## Those constants are for playing around ##########

"""Data sandbox widget values"""

LASSO_SLIDER_START = 1
LASSO_SLIDER_END = 100
LASSO_SLIDER_STEP = 1  # positive integer expected
LASSO_SLIDER_STARTING_VAL = 10


CLUSTER_RANGE_X = (-20, 20)
CLUSTER_RANGE_Y = (-20, 20)
CLUSTER_RANGE_STEP = 1


FREEHAND_DENSITY_START = 1
FREEHAND_DENSITY_END = 100
FREEHAND_DENSITY_STEP = 1  # positive integer expected
FREEHAND_DENSITY_STARTING_VAL = 10

FREEHAND_DEVIATION_START = 0
FREEHAND_DEVIATION_END = 20
FREEHAND_DEVIATION_STEP = 1
FREEHAND_DEVIATION_STARTING_VAL = 2

"""Neural options"""

NEURAL_DEF_SLIDER_STEPS = 3
NEURAL_DEF_MAX_ITER_STEPS = 100
NEURAL_DEF_ACTIVATION = 1  # 0: identity, 1: sigmoid, 2: tanh, 3: linear
NEURAL_DEF_SOLVER = 1  # 0: lbfgs, 1: gradient descent, 2: adam
NEURAL_DEF_LAYERS = "15, 15"  # more layers: "20, 10, 5, 20"
LOSS_PRINT = "log"  # either "log", "app", None

"""Regression sublayouts"""

POLY_DEF_DGR = 3

POL_FROM_DGR = 1
POL_TO_DGR = 5

"""Classification sublayouts"""

KNN_DEF_NEIGHBOUR_N = 3


CLUSTERS_COUNT_DEF = 3
CLUSTER_SIZE_DEF = 3
CLUSTER_DEV_DEF = 1  # cluster size plus minus
MAX_CLUSTERS = 10

########## Those constants are for experiments (interesting app behaviour, but possible (bigger) instability) ########

"""used in basic_sublayouts.py"""

# Sets the detail of classifier image
MESH_STEP_SIZE = 0.05  # default: 0.05

# Sets the detail of regression line
LINE_POINTS = 1000  # default 1000

X_EXT = 1
Y_EXT = X_EXT

BETA_PLOT_SAMPLES = 250
BETA_PLOT_DETAIL = 20

########## Those constants should NOT be changed ##########

# used in data_sandbox.py
DENS_INPUT_DEF_VAL = ""

# used in ClassifierLayout.py as a default point color which is immediately replaced
EMPTY_VALUE_COLOR = 'black'

# Used in DataSandbox
UNIFORM_MODE = "Uniform"
BETA_MODE = "Beta"

# used in general_layout.py
CLASS_SELECT_BUTTON_WIDTH = 100  # in pixels, default: 100
MAX_CLASS_NAME_LENGTH = 8  # default: 8

