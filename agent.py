import numpy as np
from functools import reduce
import itertools
from info_theory import blahut_arimoto
from strategy.empowerment import empowerment

def rand_sample(p_x):
    """
    Randomly sample a value from a probability distribution p_x
    """
    cumsum = np.cumsum(p_x)
    rnd = np.random.rand()
    return np.argmax(cumsum > rnd)

def normalize(X):
    """
    Normalize vector or matrix columns X
    """
    return X / X.sum(axis=0)

def softmax(x, tau):
    """
    Returns the softmax normalization of a vector x using temperature tau.
    """
    return normalize(np.exp(x / tau))

class EmpMaxAgent:
    """
    Model-based reinforcement learning agent maximising its empowerment.

    This agent uses an intrinsic reinforcement learning framework to maximise its empowerment.
    It does this by computing its own reward signal, as the empowerment with respect to the
    current learned world model.
    """

    def __init__(self, T, det, alpha=0.1, gamma=0.9, n_step=1, n_samples=500):
        """
        T : transition matrix, numpy array
            Transition matrix describing environment dynamics. T[s',a,s] is the probability of
            transitioning to state s' given you did action a in state s.
        alpha : learning rate, float
        gamma : discount factor, float
            Discounts future rewards. Fraction, between 0 and 1.
        n_step : time horizon, int
        n_samples : int
            Number of samples used in empowerment computation if environment is deterministic.
        """
        self.alpha = alpha
        self.gamma = gamma
        # transition function
        self.T = T
        self.n_s, self.n_a, _ = T.shape
        # reward function
        self.R = 5 * np.ones([self.n_s, self.n_a])
        # experience
        self.D = np.zeros([self.n_s, self.n_a, self.n_s])
        # action-value
        self.Q = 50 * np.ones([self.n_s, self.n_a])
        self.n_step = n_step
        self.n_samples = n_samples
        self.det = det
        self.a_ = None
        self.k_s = 10
        self.k_a = 3
        self.E = np.array([self.estimateE(s) for s in range(self.n_s)])
        # temperature parameter
        self.tau0 = 15  # initial tau
        self.tau = self.tau0
        self.t = 0
        self.decay = 5e-4
        self.s = None

    def act(self, s):
        if self.a_ is None:
            return rand_sample(softmax(self.Q[s, :], tau=self.update_tau()))
        else:
            return self.a_

    def update_tau(self):
        self.tau = self.tau0 * np.exp(-self.decay * self.t)
        self.t += 1
        return self.tau

    def update(self, s, a, s_, influence=0.):
        # append experience, update model
        self.D[s_, a, s] += 1
        self.T[:, a, s] = normalize(self.D[:, a, s])
        # compute reward as empowerment achieved
        r = self.estimateE(s_) + influence
        self.E[s_] = r
        # update reward R
        self.R[s, a] = self.R[s, a] + self.alpha * (r - self.R[s, a])
        # update action-value Q
        nz = np.where(self.T[:, a, s] != 0)[0]
        self.Q[s, a] = self.R[s, a] + self.gamma * (np.sum(self.T[nz, a, s] * np.max(self.Q[nz, :], axis=1)))
        # randomly update Q at other states
        for s in np.random.randint(0, self.n_s, self.k_s):
            for a in np.random.randint(0, self.n_a, self.k_a):
                self.Q[s, a] = self.R[s, a] + self.gamma * (np.sum(self.T[:, a, s] * np.max(self.Q, axis=1)))

    def estimateE(self, state):
        return empowerment(self.T, self.det, self.n_step, state, n_samples = self.n_samples)

    @property
    def action_map(self):
        return np.argmax(self.Q, axis=1)

    @property
    def value_map(self):
        return np.max(self.Q, axis=1)

