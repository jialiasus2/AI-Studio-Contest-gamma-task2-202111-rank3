

DATA_ROOT = '../competition_data/'
TRAIN_IMAGE_FOLDER = DATA_ROOT+'training/fundus color images/'
TRAIN_LABEL_PATH = DATA_ROOT+'training/fovea_localization_training_GT.xlsx'
TEST_IMAGE_FOLDER = DATA_ROOT+'testing/fundus color images/'
RESULT_PATH = './Localization_Results.csv'

INPUT_IMAGE_SHAPE = 512


OUTPUT_RANGE = (0.3, 0.7)

FEAT_INDICES = [-1]
LOSS_WEIGHTS = [1.]

BATCH_SIZE = 4
LR = 1e-3
WARMUP_EPOCH = 20
TRAIN_EPOCHS = 480
EVAL_EPOCH = 1

MODEL_PATH = '../models'
