import pandas as pd
from src.config import TRAIN_PATH, TEST_PATH

def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test
