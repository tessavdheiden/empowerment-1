from functools import reduce
import itertools
import numpy as np

class World(object):
    def act(self, s, a):
        raise NotImplementedError

    def plot(self, fig, ax):
        raise NotImplementedError

    def _index_to_cell(self, s):
        cell = [int(s / self.dims[1]), s % self.dims[1]]
        return np.array(cell)

    def _cell_to_index(self, cell):
        return cell[1] + self.dims[1]*cell[0]

    def compute_transition(self, det = 1.):
        """ Computes probabilistic model T[s',a,s] corresponding to the maze world.
        det : float between 0 and 1
            Probability of action successfully performed (otherwise a random different action is performed with probability 1 - det). When det = 1 the dynamics are deterministic.
        """
        n_actions = len(self.actions)
        n_states = self.dims[0]*self.dims[1]
        # compute environment dynamics as a matrix T
        T = np.zeros([n_states,n_actions,n_states])
        # T[s',a,s] is the probability of landing in s' given action a is taken in state s.
        for s in range(n_states):
            for i, a in enumerate(self.actions.keys()):
                s_new = self.act(s, a)
                s_unc = list(map(lambda x : self.act(s, x), filter(lambda x : x != a, self.actions.keys())))
                T[s_new, i, s] += det
                for su in s_unc:
                    T[su, i, s] += (1-det)/(len(s_unc))
        self.T = T
        return T

    def compute_nstep_transition_model(self, n_step):
        n_states, n_actions, _ = self.T.shape
        nstep_actions = np.array(list(itertools.product(range(n_actions), repeat=n_step)))
        Bn = np.zeros([n_states, len(nstep_actions), n_states])
        for i, an in enumerate(nstep_actions):
            Bn[:, i, :] = reduce((lambda x, y: np.dot(y, x)), map((lambda a: self.T[:, a, :]), an))
        self.Bn = Bn
        return Bn
