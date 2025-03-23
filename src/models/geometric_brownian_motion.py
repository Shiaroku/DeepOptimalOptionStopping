import numpy as np
from numpy.random import default_rng

class GBM():
    """ Geometric Brownian Motion """
    def __init__(self, r, mu, sigma, corr_matrix):
        self.r = r
        self.mu = mu
        self.sigma = sigma
        self.corr_matrix = corr_matrix

    def generate_trajectories(self, S0, T, N, n_paths, rng=default_rng()):
        """ Stochastic trajectories generated through Euler method """
        if np.isscalar(S0):
            return self.__gbm_1d(S0, T, N, n_paths, rng)
        d = len(S0)
        return self.__gbm_dd(S0, T, N, n_paths, d, rng)

    def __gbm_1d(self, S0, T, N, n_paths, rng):
        dt = T / (N - 1)
        drift_term = (self.mu - 0.5 * self.sigma**2) * dt
        drift_increments = drift_term * np.ones((N - 1, n_paths, 1))
        diffusion_increments = self.sigma * rng.normal(
            loc=0,
            scale=np.sqrt(dt),
            size=(N - 1, n_paths, 1)
        )
        increments = drift_increments + diffusion_increments
        initial_zeros = np.zeros((1, n_paths, 1))
        cumulated_increments = np.cumsum(
            np.concatenate([initial_zeros, increments], axis=0),
            axis=0
        )
        return S0 * np.exp(cumulated_increments)

    def __gbm_dd(self, S0, T, N, n_paths, d, rng):
        # Placeholder for multidimensional GBM
        raise NotImplementedError("Multi-dimensional GBM not implemented yet.")
