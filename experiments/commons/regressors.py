from typing import List

import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate import NearestNDInterpolator

from sklearn.model_selection import train_test_split

from experiments.commons import util

import rsqtoa
from rsqtoa import FunctionApproximator
from rsqtoa.regressors import Regressors, RsqtoaRegressor
from rsqtoa.model_wrapper import EquationModel


class ANNRegressor(Regressors):

    def __init__(self, input_dimension: int = 2, validation_fraction: float = 0.2, random_state: int = 0):
        self.__model = util.get_ann_function_approximator(input_dimension)

        Regressors.__init__(self, validation_fraction=validation_fraction, random_state=random_state)

    def fit(self, x_train, y_train):
        X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=self.validation_fraction,
                                                          random_state=self.random_state)
        self.loss_hist = rsqtoa.train_model(self.__model, X_train, Y_train, X_val, Y_val)

    def predict(self, features):
        return self.__model.predict(features)

    def plot_loss_history(self):
        plt.plot(self.loss_hist['train_loss_points'][1])
        plt.plot(self.loss_hist['val_test_loss_points'][1])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


class InterpolatorRegressor(Regressors):

    def __get_interpolator_instance(self, features, target):
        return NearestNDInterpolator(features, target, rescale=True)

    def __init__(self, config_path: str, lower: List[float], upper: List[float],
                 validation_fraction: float = 0.2, random_state: int = 0):
        self.__config_path = config_path
        self.__lower = lower
        self.__upper = upper
        self.__type = type

        Regressors.__init__(self=self, validation_fraction=validation_fraction, random_state=random_state)

    def fit(self, features, target):
        ip =  NearestNDInterpolator(features, target, rescale=True)

        def f_interpol(x):
            return float(ip([x])) if np.shape(x) == (features.shape[1],) else ip(x.T)

        eqn_model = EquationModel(f_interpol, features.shape[1], self.__lower, self.__upper)
        self.regressor = RsqtoaRegressor(config_path=self.__config_path,
                                    get_new_approximator_instance=self.create_approximator_instance,
                                    model=eqn_model)
        self.regressor.fit(features=features, target=target)
    def predict(self, features):
        return self.regressor.predict(features)

    def plot_loss_history(self):
        self.regressor.plot_loss_history()

    def create_approximator_instance(self, input_dimension: int) -> FunctionApproximator:
        return util.get_ann_function_approximator(input_dimension=input_dimension)
