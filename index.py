# import the necessary packages
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from lsh import LSH

class JenaIndex:
    def __init__(self, name, extractor):
        # store our index of images
        self.extractor = extractor
        self.name = name
        self.reset()

    def features_at(self, period):
        periods = self.jena_periods.tolist()
        idx = periods.index(period)
        return self.jena_features[idx]

    def index(self, jena_dict, reset=False):
        if reset:
            self.reset()
        jena_period, jena_features = [], []
        for period, values in jena_dict.items():
            features = self.extractor.extract(values)
            print(features.shape)
            jena_period.append(period)
            jena_features.append(features)
        if len(jena_period) > 0:
            self.jena_periods = np.array(jena_period)
            self.jena_features = np.stack(jena_features)

    def reset(self):
        self.jena_periods = None
        self.jena_features = None

    def save(self, parent_path):
        assert self.jena_periods is not None and self.jena_features is not None, "no data to save"
        parent = Path(parent_path)
        # assert parent_path.exists(), 'path should exist'

        # save names
        dates_path = parent / (self.name + '_dates' + '.npy')
        np.save(dates_path, self.jena_periods)

        # save features
        features_path = parent / (self.name + '_features' + '.npy')
        np.save(features_path, self.jena_features)

    def load(self, parent_path):
        parent = Path(parent_path)
        # assert parent.exists(), 'path should exist'
        # save names
        dates_path = parent / (self.name + '_dates' + '.npy')
        self.jena_dates_ = np.load(dates_path)
        features_path = parent / (self.name + '_features' + '.npy')
        self.jena_features = np.load(features_path)

    def search(self, query, k=1, return_distance=True):
        # initialize our dictionary of results
        results = []
        query_feature = self.extractor.extract(query)
        # loop over the features
        for i, feature in enumerate(self.jena_features):
            distance = cdist([query_feature], [feature])
            period = self.jena_periods[i]
            results.append((distance, period, i))
        results = sorted([(dist, date, idx) for dist, date, idx in results])
        # return our results
        return results[:k]

class SKLearnJenaIndex(JenaIndex):

    def __init__(self, name, extractor, algorithm='brute'):
        super().__init__(name, extractor)
        self.algorithm = NearestNeighbors(algorithm=algorithm, metric='euclidean')

    def index(self, jena_dict, reset=False):
        super().index(jena_dict, reset)
        self.algorithm.fit(self.jena_features)

    def search(self, query, k=1, return_distances=True):
        query_feature = self.extractor.extract(query)
        distances, indices = self.algorithm.kneighbors(np.array([query_feature]), k)
        periods = self.jena_periods[indices]
        if return_distances:
            return periods[:k], distances[:k]
        else:
            return periods[:k]

class LSHJenaIndex(JenaIndex):

    def __init__(self, name, extractor, algorithm='brute'):
        super().__init__(name, extractor)
        self.lsh = LSH(nb_projection=10, nb_tables=2, w=1.0)

    def index(self, jena_dict, reset=False):
        super().index(jena_dict, reset)
        self.lsh.fit(self.jena_features)

    def search(self, query, k=1, return_distances=True):
        query_feature = self.extractor.extract(query)
        distances, indices = self.lsh.kneighbors(query_feature, k=k)
        periods = self.jena_periods[indices]
        if return_distances:
            return periods[:k], distances[:k]
        else:
            return periods[:k]

class BruteForceJenaIndex(JenaIndex):
    def __init__(self, name, extractor, dist='euclidean'):
        assert dist in ["manhattan", "euclidean", "chebychev"], "unknown distances"
        super().__init__(name, extractor)
        self.dist=dist

    def search(self, query, k=1, return_distances=True):
        query_feature = self.extractor.extract(query)
        lsh_result = self._knn_search(query_feature, k=k)
        distances, indices = lsh_result
        periods = self.jena_periods[indices]
        if return_distances:
            return periods[:k], distances[:k]
        else:
            return periods[:k]

    def _knn_search(self, query_features, k=1):
        distances = self._compute_distances(self.jena_features, query_features)
        if k == 1:
            min_idx = np.argmin(distances)
            return [min_idx], [distances[min_idx]]
        else:
            min_idx = np.argpartition(distances, k)[:k]
            return min_idx, distances[min_idx]

    def _compute_distances(self, data, query):
        distances = np.zeros((len(data),), dtype=np.float32)
        if self.dist == "manhattan":
            distances = np.sum(np.abs(data - query), axis=1)
        elif self.dist == "euclidean":
            distances = np.sqrt(np.sum((data - query)**2, axis=1))
        elif self.dist == "chebychev":
            distances = np.max(np.abs(data - query), axis=1)
        return distances
