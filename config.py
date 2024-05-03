import torch

# DEVICE = torch.device('cuda:0')
DEVICE = torch.device("cpu")
DATASET_PATH = "dataset/images"
TRAIN_DATASET = "dataset/train.csv"
TEST_DATASET = "dataset/test.csv"
COMPOSED_GTREND = "dataset/gtrends.csv"
CATEG_DICT = "dataset/category_labels.pt"
COLOR_DICT = "dataset/color_labels.pt"
FAB_DICT = "dataset/fabric_labels.pt"
NUM_EPOCHS = 2
USE_TEACHERFORCING = True
TF_RATE = 0.5
LEARNING_RATE = 0.0001
NORMALIZATION_VALUES_PATH = "dataset/normalization_scale.npy"
BATCH_SIZE= 128
SHOW_PLOTS = False
NUM_WORKERS = 8
USE_EXOG = True
EXOG_NUM = 3
EXOG_LEN = 52
HIDDEN_SIZE = 300
SAVED_FEATURES_PATH = "incv3_features"
USE_SAVED_FEATURES = False
NORM = False
model_types = ["image", "concat", "residual", "cross"]
MODEL = 1
