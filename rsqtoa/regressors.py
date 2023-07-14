from abc import ABC, abstractmethod
from typing import Union, Callable

import numpy as np

import matplotlib.pyplot as plt

import rsqtoa
from rsqtoa.model_wrapper import Model


class Regressors(ABC):
    loss_hist = []

    def __init__(self, validation_fraction: float = 0.2, random_state: int = 0):
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    @abstractmethod
    def fit(self, features, target):
        pass

    @abstractmethod
    def predict(self, features):
        pass

    def plot_loss_history(self):
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1)
        for i, loss_hist in enumerate(self.loss_hist):
            ax.plot(loss_hist['train_loss_points'][0],
                    loss_hist['train_loss_points'][1],
                    label='Training Loss ({})'.format(i))
            ax.plot(loss_hist['val_test_loss_points'][0],
                    loss_hist['val_test_loss_points'][1],
                    label='Test Loss ({})'.format(i))
        ax.grid(True)
        ax.legend()
        plt.show()


class RsqtoaRegressor(Regressors):

    def __init__(self, config_path: str, model: Model,
                 get_new_approximator_instance: Callable,
                 validation_fraction: float = 0.2,
                 random_state: Union[int, None] = None):
        self._config = rsqtoa.create_config(cfg=config_path)
        self._model = model
        self._get_new_approximator_instance = get_new_approximator_instance

        Regressors.__init__(self=self, validation_fraction=validation_fraction,
                            random_state=random_state)

    def fit(self, features, target):
        self._rsqtoa_tree = rsqtoa.create_subspace_tree(self._config,
                                                        self._model)

        # Train all reduced models
        coeff, N_test, N_training = 0, self._config[
            'approximator_test_samples'], self._config[
            'approximator_train_samples']

        # Returns all subspaces
        leaves = self._rsqtoa_tree.get_leafs()
        coeff = sum(
            [1 if len(leaf.non_separable_dims) > 0 else 0 for leaf in leaves])
        self.__loss_hist = []
        for leaf in leaves:
            if len(leaf.non_separable_dims) > 0:
                # Create samples in the subspace (training & test data)
                training_data_x, training_data_y = leaf.sample_domain(
                    int(N_training / coeff))
                test_data_x, test_data_y = leaf.sample_domain(
                    int(N_test / coeff))

                # Evaluate the reduced model at the training samples
                training_data_y = leaf.evaluate_reduced_model(
                    np.transpose(training_data_x))
                training_data_y = np.expand_dims(training_data_y, axis=1)

                # Evaluate the reduced model at the test samples
                test_data_y = leaf.evaluate_reduced_model(
                    np.transpose(test_data_x))
                test_data_y = np.expand_dims(test_data_y, axis=1)

                # Normalize the data for training
                idx = rsqtoa.normalize_data(training_data_y, test_data_y)

                # Create reduced data set
                input_n = len(leaf.non_separable_dims)
                for i in reversed(range(leaf.dims)):
                    if i not in leaf.non_separable_dims:
                        training_data_x = np.delete(training_data_x, i, 1)
                        test_data_x = np.delete(test_data_x, i, 1)

                # Create tensorflow model
                approximator = self._get_new_approximator_instance(
                    input_dimension=input_n)

                # Train the tensorflow model
                loss_hist = rsqtoa.train_model(
                    approximator, training_data_x, training_data_y, test_data_x,
                    test_data_y)
                self.__loss_hist.append(loss_hist)

                # Set the trained tensorflow model in the subspace node
                leaf.approximated_model = approximator

                # Set the nomalization index so taht we can denormalize the results
                leaf.denormalization_idx = idx

        # Print some debug info
        self._rsqtoa_tree.print_info()

    def predict(self, features):
        return self._rsqtoa_tree.evaluate_reassembled_model(features.T)

    def plot_loss_history(self):
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1)
        for i, loss_hist in enumerate(self.__loss_hist):
            ax.plot(loss_hist['train_loss_points'][0],
                    loss_hist['train_loss_points'][1],
                    label='Training Loss ({})'.format(i))
            ax.plot(loss_hist['val_test_loss_points'][0],
                    loss_hist['val_test_loss_points'][1],
                    label='Test Loss ({})'.format(i))
        ax.grid(True)
        ax.legend()
        plt.show()
