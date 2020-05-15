import numpy as np
import matplotlib.pyplot as plt
import torch

class EmpowermentStrategy(object):
    def compute(self, world, T, det, n_step):
        """
        Compute the empowerment of a state in a grid world
        T : numpy array, shape (n_states, n_actions, n_states)
            Transition matrix describing the probabilistic dynamics of a markov decision process
            (without rewards). Taking action a in state s, T describes a probability distribution
            over the resulting state as T[:,a,s]. In other words, T[s',a,s] is the probability of
            landing in state s' after taking action a in state s. The indices may seem "backwards"
            because this allows for convenient matrix multiplication.
        det : bool
            True if the dynamics are deterministic.
        n_step : int
            Determines the "time horizon" of the empowerment computation. The computed empowerment is
            the influence the agent has on the future over an n_step time horizon.
        n_samples : int
            Number of samples for approximating the empowerment in the deterministic case.
        state : int
            State for which to compute the empowerment.
        """
        raise NotImplementedError

    @property
    def action_map(self):
        return np.argmax(self.q_x, axis=0)

    def plot(self, width, height):
        Amap = self.action_map
        n_states = width * height
        assert self.q_x.shape[1] == n_states
        assert np.sum(np.isnan(self.q_x)) == (n_states*self.q_x.shape[0]) or (all(np.sum(self.q_x, axis=0) > .9) and all(np.sum(self.q_x, axis=0) < 1.1))
        prob = np.max(self.q_x, axis=0)
        if not np.sum(np.isnan(self.q_x)) == (n_states*self.q_x.shape[0]):
            U = np.array([(1 if a == 2 else (-1 if a == 3 else 0)) for a in Amap]).reshape(height, width)
            V = np.array([(1 if a == 0 else (-1 if a == 1 else 0)) for a in Amap]).reshape(height, width)
            plt.quiver(np.arange(width) + .5, np.arange(height) + .5, U * prob.reshape(height, width), V * prob.reshape(height, width))