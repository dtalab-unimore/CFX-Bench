import pandas as pd


class DatasetLoader:
    def __init__(self, features, categorical_features, continuous_features, data_orig, data_bin, name='dataset'):
        self.columns = {name: features}
        self.categorical_features = {name: categorical_features}
        self.continuous_features = {name: continuous_features}
        self.name = name
        self.features_tree = {}
        self.data_orig = data_orig
        self.data_bin = data_bin

        data_oh, features = [], []
        for x in self.data_orig.columns[:-1]:
            self.features_tree[x] = []
            for y in self.data_bin.columns[:-1]:
                if y.startswith(x):
                    data_oh.append(data_bin[y])
                    feature_value = x + " = " + y.removeprefix(x + "_")
                    features.append(feature_value)
                    self.features_tree[x].append(feature_value)
        self.data_oh = pd.concat(data_oh, axis=1, ignore_index=True)
        self.data_oh.columns = features
        self.features = features
        self.features.append(data_orig.columns[-1])
        self.data = pd.concat([self.data_oh, data_orig[data_orig.columns[-1]]], axis=1)