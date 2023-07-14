import numpy as np

import numpy as np


class PlotFormatter:
    def __init__(self, x_ranges_endpoints, tf_savedModel, ground_truth_function = None, resolution_per_dim = 100, alpha=1.):

        # this can now be any kind of model!
        self.model = tf_savedModel
        self.alpha = alpha # for opacity
        if len(x_ranges_endpoints) !=2 or np.stack(x_ranges_endpoints).shape[1] != 2:
            raise Exception("2D plot means that there should be 2 ranges. Provide [[x1_start, x1_end], [x2_start, x2_end]]")
        self.x_ranges_endpoints = x_ranges_endpoints
        self.resolution_per_dim = resolution_per_dim
        self.ground_truth_function = ground_truth_function

    def create_meshgrid(self):
        # build a 2D-meshgrid! list [(rpd, rpd), (rpd, rpd)] with
        linspaces = [np.linspace(start, end, self.resolution_per_dim) for [start, end] in self.x_ranges_endpoints]
        meshgrid_2D = np.stack(np.meshgrid(*linspaces, indexing='ij'))
        return meshgrid_2D

    def plot_on_2d_meshgrid(self, function, meshgrid_2D):
        # meshgrid2D will be of shape (2, n, m). We want each meshgrid2D[:,i,j] to represent a point => points_as_input[q]
        points_as_input = np.reshape(meshgrid_2D, [2, -1])  # flatten last two dimensions
        points_as_input = np.transpose(points_as_input, [1, 0])
        import tensorflow as tf
        print(points_as_input.shape)
        ys = function(points_as_input.astype(np.float64))

        # we need the points to be in the original format.
        shape_2d = meshgrid_2D[0].shape
        ys_2D = np.reshape(ys, shape_2d)

        return ys_2D

    def plot_model(self, axToPlotOn, label='prediction\n'):
        if self.ground_truth_function == None:
            raise Exception("Need to be initialized with a ground-truth-function")

        meshgrid_2D = self.create_meshgrid()
        ys_2D = self.plot_on_2d_meshgrid(self.model, meshgrid_2D)
        # now plot ys_2d on the two linspaces
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        axToPlotOn.plot_surface(meshgrid_2D[0], meshgrid_2D[1], ys_2D, alpha = self.alpha)
        axToPlotOn.set_title(label)
        axToPlotOn.set_xlabel(r'$x_1$')
        axToPlotOn.set_ylabel(r'$x_2$')


    def plot_ground_truth(self, axToPlotOn, label='ground truth\n'):
        meshgrid_2D = self.create_meshgrid()
        ys_2D = self.plot_on_2d_meshgrid(self.ground_truth_function, meshgrid_2D)
        axToPlotOn.plot_surface(meshgrid_2D[0], meshgrid_2D[1], ys_2D, alpha = self.alpha)
        axToPlotOn.set_title(label)
        axToPlotOn.set_xlabel(r'$x_1$')
        axToPlotOn.set_ylabel(r'$x_2$')


