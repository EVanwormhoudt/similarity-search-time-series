import numpy as np
from pandas import DataFrame
from typing import List

class JenaPeriodExtractor:

    def __init__(self, features: List[str]):
        self._features = features

    def extract(self, data: DataFrame):
        feature_arrays = []
        for feature in self._features:
            feature_values = data[feature].values
            feature_arrays.append(np.array(feature_values))
        return np.concatenate(feature_arrays)

class JenaExtractor:

    def __init__(self, features: List[str] = None):
        self._features = features

    def extract(self, dataframe: DataFrame):
        sub_df = dataframe[self._features] if self._features else dataframe
        values = sub_df.values
        return values.flatten('C')