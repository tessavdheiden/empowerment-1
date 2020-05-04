""" 
Module allowing for the computation of n-step empowerment given a matrix 
describing the probabilistic dynamics of an environment.
"""

import numpy as np 
from functools import reduce
import itertools
from empowerment_strategy import EmpowermentStrategy
from info_theory import blahut_arimoto, _rand_dist, blahut_arimoto_batched


def _normalize_mat(P):
    row_sums = P.sum(axis=1)
    return P / row_sums[:, np.newaxis]


class BlahutArimoto(EmpowermentStrategy):
    def compute(self, world, T, n_step, epsilon=1e-6):
        """
        Compute the empowerment of a state in a grid world
        epsilon : float
            Value that terminates the optimization for approximating the empowerment in the probablistic case.
        """
        n_states, n_actions, _ = T.shape
        nstep_actions = list(itertools.product(range(n_actions), repeat=n_step))
        Bn = np.zeros([n_states, len(nstep_actions), n_states])
        q_x = _rand_dist((Bn.shape[1],))
        self.q_x = _normalize_mat(np.repeat(np.expand_dims(q_x, axis=1), n_states, axis=1))
        for i, an in enumerate(nstep_actions):
            Bn[:, i, :] = reduce((lambda x, y: np.dot(y, x)), map((lambda a: T[:, a, :]), an))

        E = np.zeros(world.dims)
        for y in range(world.dims[0]):
            for x in range(world.dims[1]):
                idx = int(y*world.dims[1] + x)
                s = world._cell_to_index((y, x))
                E[y, x] = blahut_arimoto(Bn[:, :, s], self.q_x[:, idx], epsilon=epsilon)

        return E


class VisitCount(EmpowermentStrategy):
    def compute(self, world, T, n_step, n_samples=1000):
        """
        Compute the empowerment of a state in a grid world
        n_samples : int
            Number of samples for approximating the empowerment in the deterministic case.
        """
        n_states, n_actions, _  = T.shape
        # only sample if too many actions sequences to iterate through
        if n_actions**n_step < 5000:
            nstep_samples = np.array(list(itertools.product(range(n_actions), repeat = n_step)))
        else:
            nstep_samples = np.random.randint(0,n_actions, [n_samples,n_step])
        self.q_x = _normalize_mat(np.random.rand(n_actions**n_step, n_states))
        # fold over each nstep actions, get unique end states
        tmap = lambda s, a: np.argmax(T[:, a, s])

        E = np.zeros(world.dims)
        for y in range(world.dims[0]):
            for x in range(world.dims[1]):
                s = world._cell_to_index((y, x))
                seen = set()
                for i in range(len(nstep_samples)):
                    aseq = nstep_samples[i,:]
                    seen.add(reduce(tmap, [s,*aseq]))
                E[y, x] = np.log2(len(seen)) # empowerment = log # of reachable states
        return E


class BlahutArimotoTimeOptimal(EmpowermentStrategy):
    def compute(self, world, T, n_step, epsilon=1e-6):
        n_states, n_actions, _ = T.shape
        nstep_actions = list(itertools.product(range(n_actions), repeat=n_step))
        Bn = np.zeros([n_states, len(nstep_actions), n_states])
        q_x = _rand_dist((Bn.shape[1],))
        for i, an in enumerate(nstep_actions):
            Bn[:, i, :] = reduce((lambda x, y: np.dot(y, x)), map((lambda a: T[:, a, :]), an))

        self.q_x = _normalize_mat(np.repeat(np.expand_dims(q_x, axis=1), n_states, axis=1))
        return blahut_arimoto_batched(Bn, self.q_x, epsilon=epsilon).reshape(world.dims)


class VisitCountFast(EmpowermentStrategy):
    def compute(self, world, T, n_step, n_samples=1000):
        n_states, n_actions, _ = T.shape
        # only sample if too many actions sequences to iterate through
        if n_actions ** n_step < 5000:
            nstep_actions = np.array(list(itertools.product(range(n_actions), repeat=n_step)))
        else:
            nstep_actions = np.random.randint(0, n_actions, [n_samples, n_step])
        self.q_x = _normalize_mat(np.random.rand(n_actions ** n_step, n_states))
        Bn = np.zeros([n_states, len(nstep_actions), n_states])
        for i, an in enumerate(nstep_actions):
            Bn[:, i, :] = reduce((lambda x, y: np.dot(y, x)), map((lambda a: T[:, a, :]), an))
        # fold over each nstep actions, get unique end states
        seen = map(lambda x: np.unique(x), np.argmax(Bn[:, :, :], axis=0).T)
        return np.fromiter(map(lambda x: np.log2(len(x)), seen), dtype=np.float).reshape(world.dims)



