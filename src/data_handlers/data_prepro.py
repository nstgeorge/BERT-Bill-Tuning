import pickle
import numpy as np
import pandas as pd
import argparse
import os

class DataPreprocessor:

    def __init__(self, path):
        self.load_file(path)

    def load_file(self, path):
        self.data = pd.DataFrame(pd.read_pickle(open(path, "rb"))).T

    def remove_all_empty(self):
        self.data.replace("", np.nan, inplace=True)
        self.data.dropna(inplace=True)

    def drop_subject_below_count(self, limit):
        for sub, count in self.data.subject.value_counts(sort=False).iteritems():
            if count < limit:
                self.data = self.data[self.data.subject != sub]

    def get_data(self):
        return self.data

if __name__ == "__main__":

    # Set up argparser
    # parser = argparse.ArgumentParser(description="Preprocess the given pickle file containing raw data for training in BERT.")
    # parser.add_argument("path", help="Path to pickle file")

    # args = parser.parse_args()

    dp = DataPreprocessor("raw_data/dataset.p")
    dp.remove_all_empty()
    dp.drop_subject_below_count(45)
    data = dp.get_data()

    print("Data sample: ")
    print(data[:20])

    print("\nData shape: ")
    print(data.shape)
    
    print("\nPossible subjects ({}): ".format(len(data.subject.unique())))
    print(data.subject.value_counts())
    
    if not os.path.exists('clean_data'):
        os.makedirs('clean_data')

    pickle.dump(data, open("clean_data/transformer_ready_data_1229.p", "wb"))


