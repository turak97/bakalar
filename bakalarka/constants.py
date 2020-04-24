
"""Those constants sets common behaviour (name of saved files, ...)"""
SAVED_DATASETS_DIR_NAME = "saved_datasets"
SAVED_DATASET_FILE_NAME = "my_dataset.csv"

"""Those constants are for playing around"""

NEURAL_DEF_SLIDER_STEPS = 3
NEURAL_DEF_MAX_ITER_STEPS = 100
NEURAL_DEF_ACTIVATION = 1  # 0: identity, 1: sigmoid, 2: tanh, 3: linear
NEURAL_DEF_SOLVER = 1  # 0: lbfgs, 1: gradient descent, 2: adam
NEURAL_DEF_LAYERS = "20"  # more layers: "20, 10, 5, 20"

CLUSTERS_COUNT_DEF = 3
CLUSTER_SIZE_DEF = 3
CLUSTER_VOL_DEF = 1  # cluster size plus minus
MAX_CLUSTERS = 10

"""Those constants are for experiments (interesting app behaviour, but possible (bigger) instability)"""
# used in basic_sublayouts.py Sets the detail of classifier image
MESH_STEP_SIZE = 0.05  # default: 0.05
X_EXT = 2
Y_EXT = X_EXT

POL_FROM_DGR = 1
POL_TO_DGR = 10

"""Those constants should NOT be changed"""
# used in data_sandbox.py
DENS_INPUT_DEF_VAL = ""

# used in ClassifierLayout.py as a default point color which is immediately replaced
EMPTY_VALUE_COLOR = 'black'

# used in general_layout.py
CLASS_SELECT_BUTTON_WIDTH = 100  # in pixels, default: 100
MAX_CLASS_NAME_LENGTH = 8  # default: 8

