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
