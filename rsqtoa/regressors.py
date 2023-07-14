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


# class ANNRegressor(Regressors):
#
#     def __init__(self, input_dimension: int = 2, validation_fraction: float = 0.2, random_state: int = 0):
#         self.__model = util.get_ann_function_approximator(input_dimension)
#
#         Regressors.__init__(self, validation_fraction=validation_fraction, random_state=random_state)
#
#     def fit(self, x_train, y_train):
#         X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=self.validation_fraction,
#                                                           random_state=self.random_state)
#         self.__loss_hist = rsqtoa.train_model(self.__model, X_train, Y_train, X_val, Y_val)
#         self.__plot_loss_history()
#
#     def predict(self, features):
#         return self.__model.predict(features)
#
#     def loss_hist(self):
#         return self.__loss_hist
#
#     def __plot_loss_history(self):
#         plt.plot(self.__loss_hist['train_loss_points'][1])
#         plt.plot(self.__loss_hist['val_test_loss_points'][1])
#         plt.title('Model loss')
#         plt.ylabel('Loss')
#         plt.xlabel('Epoch')
#         plt.legend(['Train', 'Test'], loc='upper left')
#         plt.show()


# class RSQTOARegressor(Regressors):
#
#     def __init__(self, config: dict, lower: List[float], upper: List[float], partitions: int = 5, threshold: int = 5,
#                  validation_fraction: float = 0.2, random_state: int = 0):
#         self.__config = config
#         self.__lower = lower
#         self.__upper = upper
#         self.__partitions = partitions
#         self.__threshold = threshold
#
#         Regressors.__init__(self=self, validation_fraction=validation_fraction, random_state=random_state)
#
#     def fit(self, features, target):
#         grid_model = DataGridModel(features=features, target=target, lower=self.__lower, upper=self.__upper,
#                                    partitions=self.__partitions, threshold=self.__threshold)
#
#         start = time.time()
#         self.__model = rsqtoa.create_subspace_tree(self.__config, grid_model)
#         time_1 = time.time() - start
#
#         # Train all reduced models
#         N_training = 20000
#         N_test = 4000
#         # N_training = 7000
#         # N_test = 650
#         leafs = self.__model.get_leafs()  # Returns all subspaces
#         coeff = 0
#         for leaf in leafs:
#             if len(leaf.non_separable_dims) > 0:
#                 coeff += 1
#
#         self.__loss_hist = []
#         start = time.time()
#         for leaf in leafs:
#             if len(leaf.non_separable_dims) > 0:
#                 # Create samples in the subspace (training & test data)
#                 training_data_x, training_data_y = leaf.sample_domain(int(N_training / coeff))
#                 test_data_x, test_data_y = leaf.sample_domain(int(N_test / coeff))
#
#                 # Evaluate the reduced model at the training samples
#                 training_data_y = leaf.evaluate_reduced_model(np.transpose(training_data_x))
#                 training_data_y = np.expand_dims(training_data_y, axis=1)
#
#                 # Evaluate the reduced model at the test samples
#                 test_data_y = leaf.evaluate_reduced_model(np.transpose(test_data_x))
#                 test_data_y = np.expand_dims(test_data_y, axis=1)
#
#                 # Normalize the data for training
#                 idx = rsqtoa.normalize_data(training_data_y, test_data_y)
#
#                 # Create reduced data set
#                 input_n = len(leaf.non_separable_dims)
#                 for i in reversed(range(leaf.dims)):
#                     if i not in leaf.non_separable_dims:
#                         training_data_x = np.delete(training_data_x, i, 1)
#                         test_data_x = np.delete(test_data_x, i, 1)
#
#                 # Create tensorflow model
#                 model = util.get_ann_function_approximator(input_dimension=input_n)
#
#                 start = time.time()
#                 # Train the tensorflow model
#                 loss_hist = rsqtoa.train_model(
#                     model, training_data_x, training_data_y, test_data_x, test_data_y)
#                 self.__loss_hist.append(loss_hist)
#
#                 # Set the trained tensorflow model in the subspace node
#                 leaf.approximated_model = model
#
#                 # Set the nomalization index so taht we can denormalize the results
#                 leaf.denormalization_idx = idx
#
#         time_2 = time.time() - start
#         print('Real Training Time :: ', time_1 + time_2)
#
#         # Print some debug info
#         self.__model.print_info()
#         self.__plot_loss_history()
#
#     def predict(self, features):
#         return self.__model.evaluate_reassembled_model(features.T)
#
#     def loss_hist(self):
#         return self.__loss_hist[0]
#
#     def __plot_loss_history(self):
#         plt.rcParams.update({'font.size': 20})
#         fig = plt.figure(figsize=plt.figaspect(0.5))
#
#         ax = fig.add_subplot(1, 2, 1)
#         for i, loss_hist in enumerate(self.__loss_hist):
#             ax.plot(loss_hist['train_loss_points'][0], loss_hist['train_loss_points'][1],
#                     label='Training Loss ({})'.format(i))
#             ax.plot(loss_hist['val_test_loss_points'][0], loss_hist['val_test_loss_points'][1],
#                     label='Test Loss ({})'.format(i))
#         ax.grid(True)
#         ax.legend()
#         plt.show()


# class InterpolatorType(Enum):
#     NEAREST_NEIGHBOUR = 1
#     RADIAL_BASIS_FUNCTION = 2


# class InterpolatorRegressor(Regressors):
#
#     def __get_interpolator_instance(self, features, target):
#         return NearestNDInterpolator(features, target, rescale=True)
#
#     def __init__(self, config: dict, lower: List[float], upper: List[float],
#                  type: InterpolatorType = InterpolatorType.NEAREST_NEIGHBOUR,
#                  validation_fraction: float = 0.2, random_state: int = 0):
#         self.__config = config
#         self.__lower = lower
#         self.__upper = upper
#         self.__type = type
#
#         Regressors.__init__(self=self, validation_fraction=validation_fraction, random_state=random_state)
#
#     def fit(self, features, target):
#         ip = self.__get_interpolator_instance(features=features, target=target)
#
#         def f_interpol(x):
#             return float(ip([x])) if np.shape(x) == (features.shape[1],) else ip(x.T)
#
#         eqn_model = EquationModel(f_interpol, features.shape[1], self.__lower, self.__upper)
#
#         start = time.time()
#         self.__model = rsqtoa.create_subspace_tree(self.__config, eqn_model)
#         time_1 = time.time() - start
#
#         # Train all reduced models
#         # N_training = 7000
#         # N_test = 650
#         N_training = 20000
#         N_test = 4000
#         # N_training = 9500
#         # N_test = 500
#         leafs = self.__model.get_leafs()  # Returns all subspaces
#         coeff = 0
#         for leaf in leafs:
#             if len(leaf.non_separable_dims) > 0:
#                 coeff += 1
#
#         self.__loss_hist = []
#         start = time.time()
#         for leaf in leafs:
#             if len(leaf.non_separable_dims) > 0:
#                 # Create samples in the subspace (training & test data)
#                 training_data_x, training_data_y = leaf.sample_domain(int(N_training / coeff))
#                 test_data_x, test_data_y = leaf.sample_domain(int(N_test / coeff))
#
#                 # Evaluate the reduced model at the training samples
#                 training_data_y = leaf.evaluate_reduced_model(np.transpose(training_data_x))
#                 training_data_y = np.expand_dims(training_data_y, axis=1)
#
#                 # Evaluate the reduced model at the test samples
#                 test_data_y = leaf.evaluate_reduced_model(np.transpose(test_data_x))
#                 test_data_y = np.expand_dims(test_data_y, axis=1)
#
#                 # Normalize the data for training
#                 idx = rsqtoa.normalize_data(training_data_y, test_data_y)
#
#                 # Create reduced data set
#                 input_n = len(leaf.non_separable_dims)
#                 for i in reversed(range(leaf.dims)):
#                     if i not in leaf.non_separable_dims:
#                         training_data_x = np.delete(training_data_x, i, 1)
#                         test_data_x = np.delete(test_data_x, i, 1)
#
#                 # Create tensorflow model
#                 model = util.get_ann_function_approximator(input_dimension=input_n)
#
#                 start = time.time()
#                 # Train the tensorflow model
#                 loss_hist = rsqtoa.train_model(
#                     model, training_data_x, training_data_y, test_data_x, test_data_y)
#                 self.__loss_hist.append(loss_hist)
#
#                 # Set the trained tensorflow model in the subspace node
#                 leaf.approximated_model = model
#
#                 # Set the nomalization index so taht we can denormalize the results
#                 leaf.denormalization_idx = idx
#
#         time_2 = time.time() - start
#         print('Real Training Time :: ', time_1 + time_2)
#
#         # Print some debug info
#         self.__model.print_info()
#         self.__plot_loss_history()
#
#     def predict(self, features):
#         return self.__model.evaluate_reassembled_model(features.T)
#
#     def loss_hist(self):
#         return self.__loss_hist[0]
#
#     def __plot_loss_history(self):
#         plt.rcParams.update({'font.size': 20})
#         fig = plt.figure(figsize=plt.figaspect(0.5))
#
#         ax = fig.add_subplot(1, 2, 1)
#         for i, loss_hist in enumerate(self.__loss_hist):
#             ax.plot(loss_hist['train_loss_points'][0], loss_hist['train_loss_points'][1],
#                     label='Training Loss ({})'.format(i))
#             ax.plot(loss_hist['val_test_loss_points'][0], loss_hist['val_test_loss_points'][1],
#                     label='Test Loss ({})'.format(i))
#         ax.grid(True)
#         ax.legend()
#         plt.show()


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
