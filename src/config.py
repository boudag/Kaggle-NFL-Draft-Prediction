import os

DATA_DIR = '/mnt/d/asu/GCI/competition_data/competition/input'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
SUBMISSION_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

RANDOM_STATE = 42
N_FOLDS = 10
