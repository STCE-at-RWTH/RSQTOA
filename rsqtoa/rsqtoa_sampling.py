import numpy as np
from pyDOE2 import lhs

# Import abstract parent class
from abc import ABC, abstractmethod

# Import global config and statistics
from rsqtoa import rsqtoa_config as config
from rsqtoa import rsqtoa_statistics as stats
from rsqtoa.model_wrapper import Model, DataGridModel


class Sampler(ABC):
    def __init__(self, model: Model, dims, lower, upper):
        """ Constructor of the sampler base """
        self._model = model
        self._dims = dims
        self._lower = lower.astype(float)
        self._upper = upper.astype(float)
        self._samples = None

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        self._samples = samples

    @samples.deleter
    def samples(self):
        del self._samples

    @property
    def model(self) -> Model:
        return self._model

    @property
    def dims(self):
        return self._dims

    @property
    def upper(self):
        return self._upper

    @property
    def lower(self):
        return self._lower

    @abstractmethod
    def sample_dimension(self, n, dim, anchor):
        pass

    @abstractmethod
    def sample_domain(self, n, dim=-1):
        pass


# ---------------------------Domain Samplers---------------------------------- #

class DomainSampler(Sampler):
    def __init__(self, model, dims, lower, upper):
        Sampler.__init__(self, model, dims, lower, upper)

    def sample_dimension(self, n, dim, anchor):
        """ Sample in one dimension from an anchor in the domain"""
        X = np.empty((n, self.dims))

        # Broadcast anchor point and sample in the given dimension
        X[:] = np.copy(anchor)
        X[:, dim] = self.lower[dim]
        X[:, dim] += np.random.rand(n) * (self.upper[dim] - self.lower[dim])

        self.samples = np.copy(X)
        return X, self.eval_samples()

    def eval_samples(self):
        """ Evaluate the current samples set """
        if self.model is not None and self.samples is not None:
            n = np.shape(self.samples)[0]
            Y = np.empty(n)
            for i in range(n):
                Y[i] = self.model.get_target(self.samples[i])
        else:
            raise RuntimeError("Either model or samples are not defined.")

        # Increase amount of samples
        stats.amount_of_samples += n

        return Y

    @abstractmethod
    def sample_domain(self, n, dim=-1):
        pass


class RandomSampler(DomainSampler):
    def __init__(self, model, dims, lower, upper):
        """ Constructor of the random sampler """
        DomainSampler.__init__(self, model, dims, lower, upper)

    def sample_domain(self, n, dim=-1):
        """ Return n random samples. """
        X = np.random.rand(n, self.dims)
        X = self.lower + np.multiply(X, self.upper - self.lower)

        self.samples = np.copy(X)
        return X, self.eval_samples()


class LatinHypercubeSampler(DomainSampler):
    def __init__(self, model, dims, lower, upper):
        """ Constructor of the latin hypercube sampler """
        DomainSampler.__init__(self, model, dims, lower, upper)

    def sample_domain(self, n, dim=-1):
        """ Return n latin-hypercube samples. """
        if config.sampling_strategy == 'latin':
            criterion = None
        elif config.sampling_strategy == 'latin_maximin':
            criterion = 'maximin'
        elif config.sampling_strategy == 'latin_corr':
            criterion = 'correlation'
        else:
            raise ValueError(
                'Invalid sampling strategy: {}'.format(
                    config.sampling_strategy))

        X = lhs(self.dims, samples=n, criterion=criterion)
        X = self.lower + np.multiply(X, self.upper - self.lower)

        self.samples = np.copy(X)
        return X, self.eval_samples()


# ----------------------------Data Samplers----------------------------------- #

class DataSampler(Sampler):
    def __init__(self, model, dims, lower, upper):
        Sampler.__init__(self, model, dims, lower, upper)


class GridSampler(DataSampler):
    def __init__(self, model, dims, lower, upper):
        if isinstance(model, DataGridModel):
            Sampler.__init__(self, model, dims, lower, upper)
        else:
            raise Exception('Model must be of type DataGridModel')

    def sample_dimension(self, n, dim, anchor):
        return self.model.get_dimension_samples(dim, anchor, n)

    def sample_domain(self, n, dim=-1):
        return self.model.get_domain_samples(n, dim)
