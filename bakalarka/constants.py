
"""Those constants sets common behaviour (name of saved files, ...)"""
SAVED_DATASETS_DIR_NAME = "saved_datasets"
SAVED_DATASET_FILE_NAME = "my_dataset.csv"

"""Those constants are for playing around"""
POL_FROM_DGR = 1
POL_TO_DGR = 10

INIT_CLASSES_COUNT = 4  # only used when generating data while application init

NEURAL_DEF_SLIDER_STEPS = 3
NEURAL_DEF_MAX_ITER_STEPS = 100
NEURAL_DEF_ACTIVATION = 2  # 0: identity, 1: sigmoid, 2: tanh, 3: linear
NEURAL_DEF_LAYERS = "20"  # more layers: "20, 10, 5, 20"

"""Those constants are for experiments (interesting app behaviour, but possible (bigger) instability)"""
# used in ClassifierSubLayout.py Sets the detail of classifier image
MESH_STEP_SIZE = 0.05  # default: 0.05
X_EXT = 2
Y_EXT = 0.3

"""Those constants should NOT be changed"""
# used in DataSandbox.py
DENS_INPUT_DEF_VAL = ""
CLUSTER_SIZE_DEF = 8
CLUSTER_VOL_DEF = 3  # cluster size plus minus
CLUSTERS_COUNT_DEF = 4
MAX_CLUSTERS = 10

# used in Layout.py
CLASS_SELECT_BUTTON_WIDTH = 100  # in pixels, default: 100
MAX_CLASS_NAME_LENGTH = 8  # default: 8

