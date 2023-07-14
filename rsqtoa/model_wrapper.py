import random
import itertools
from typing import List, Union
from collections import Counter
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray, ArrayLike


class Model(ABC):
    def __init__(self, dims: int, lower: List, upper: List):
        self.__dims = dims
        self.__lower = lower
        self.__upper = upper

    @property
    def lower(self) -> List[Union[float, int]]:
        return self.__lower

    @property
    def upper(self) -> List[Union[float, int]]:
        return self.__upper

    @property
    def dims(self) -> int:
        return self.__dims

    @abstractmethod
    def get_target(self, feature: List[List[Union[float, int]]]) -> Union[float, int]:
        pass


class EquationModel(Model):

    def __init__(self, equation, dims, lower, upper):
        self.__equation = equation
        Model.__init__(self, dims, lower, upper)

    @property
    def equation(self):
        return self.__equation

    def get_target(self, feature: List[List[Union[float, int]]]) -> Union[float, int]:
        return self.equation(feature)


class DataGridModel(Model):
    __partitions = 5
    __threshold = 10

    def __init__(self, features: NDArray, target: ArrayLike, lower: ArrayLike, upper: ArrayLike, partitions: int = 5,
                 threshold: int = 10):
        self.__features = features
        self.__target = target
        self.__partitions = partitions
        self.__threshold = threshold

        self.__grid = self.__prepare_grid()
        self.__valid_domain_sample_indices = self.__get_valid_domain_sample_indices(self.__grid)

        Model.__init__(self, features.shape[1], lower, upper)

    @property
    def features(self) -> NDArray:
        return self.__features

    @property
    def target(self) -> ArrayLike:
        return self.__target

    def __prepare_grid(self):
        grids = {}
        for col_idx in range(self.features.shape[1]):
            grids[col_idx] = {}
            for projection_column in set(range(self.features.shape[1])).copy().difference({col_idx}):
                min, max = self.features[:, projection_column].min(axis=0), self.features[:, projection_column].max(
                    axis=0)
                min -= 1e-3
                max += 1e-3
                grids[col_idx][projection_column] = [min + (i * (max - min) / self.__partitions) for i in
                                                     range(self.__partitions + 1)]

        for voi in grids:
            mesh = np.meshgrid(*[grids[voi][var] for var in grids[voi].keys()])
            coordinates = list(np.reshape(list(zip(*[x.flat for x in mesh])), np.roll(np.shape(mesh), -1)))
            rolled_coordinates = np.roll(coordinates, np.full(np.shape(coordinates)[-1], -1),
                                         axis=list(range(np.shape(coordinates)[-1])))
            diag_coordinates = np.concatenate([coordinates, rolled_coordinates], axis=np.shape(coordinates)[-1])

            filtered_coordinates = diag_coordinates.copy()
            for i in range(len(np.shape(filtered_coordinates)) - 1):
                filtered_coordinates = np.delete(filtered_coordinates, -1, axis=i)
            filtered_coordinates = filtered_coordinates.reshape(-1, filtered_coordinates.shape[-1])
            grids[voi]['bounds'] = filtered_coordinates

        for col_idx in range(self.features.shape[1]):
            print(col_idx)

            features_subset = np.delete(self.features, [col_idx], axis=1)
            bounds = np.array(grids[col_idx]['bounds'])

            mapping = {}
            valid_elements = []

            for index, bound in enumerate(bounds):

                allInBounds = None
                for i in range(features_subset.shape[1]):
                    eval = ((features_subset[:, i] >= bound[i]) & (
                            features_subset[:, i] <= bound[i + features_subset.shape[1]]))
                    allInBounds = allInBounds & eval if allInBounds is not None else eval

                elements = np.argwhere(allInBounds)

                if len(elements) >= self.__threshold:
                    rows = list(itertools.chain.from_iterable(elements))
                    valid_elements.extend(rows)
                    mapping[tuple(bound)] = rows

            grids[col_idx]['mapping'] = mapping
            grids[col_idx]['valid_elements'] = list(set(valid_elements))

        return grids

    def __get_valid_domain_sample_indices(self, grids):
        samples = []
        for dim in grids:
            samples.append(list(itertools.chain(*[indices for indices in grids[dim]['mapping'].values()])))
        return list(set.intersection(*map(set, samples)))

    def get_domain_samples(self, num_samples: int, dimension: int = -1):
        domain_samples = random.sample(range(len(self.features)), num_samples)
        if dimension != -1:
            valid_element_indices = self.__grid[dimension]['valid_elements']
            domain_samples = random.sample(valid_element_indices, num_samples)
        return self.features[domain_samples], self.target[domain_samples]

    def get_dimension_samples(self, dimension, anchor_point, num_samples: int):
        anchor_index = Counter(np.where(self.features == anchor_point)[0]).most_common(1)[0][0]
        sample_indices = []
        sample_grid_section_anchor = []
        for key, indices in self.__grid[dimension]['mapping'].items():
            if anchor_index in set(indices):
                sample_grid_section_anchor = [np.mean([key[i], key[i + (int(len(key) / 2))]]) for i in
                                              range(int(len(key) / 2))]
                sample_indices = random.sample(indices, num_samples)
                break
        feature_samples = self.features[sample_indices]
        feature_samples[:, [i for i in range(self.dims) if i != dimension]] = sample_grid_section_anchor
        return feature_samples, self.target[sample_indices]

    def get_target(self, features: List[List[Union[float, int]]]) -> Union[float, int]:
        if any(isinstance(el, list) | (type(el) is np.ndarray) for el in features):
            targets = []
            for feature in features.T:
                targets.append(self.target[Counter(np.where(self.features == feature)[0]).most_common(1)[0][0]])
            return targets
        else:
            return self.target[Counter(np.where(self.features == features)[0]).most_common(1)[0][0]]
