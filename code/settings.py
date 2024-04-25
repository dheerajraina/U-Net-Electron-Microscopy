import os

DATA_PATH = os.path.join("../datasets/")
OUT_PATH = os.path.join("./output/")
INFERENCE_FILENAME = "electron_microscopy.keras"


# Using Adam optimizer
LEARNING_RATE = 0.0001  # 0.00005

FEATURE_MAPS = 16
CROP_DIM = 256  # Crop height and width to this size
SEED = 816      # Random seed
TRAIN_TEST_SPLIT = 0.80  # The train/test split

NUM_CHANNELS_OUT = 1
USE_DROPOUT = True  # Use dropout in model
DROPOUT_RATE = 0.1
