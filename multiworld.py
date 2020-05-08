import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Agent(object):
    def __init__(self, position, action):
        self.postion = position
        self.action = action


from mazeworld import MazeWorld, WorldFactory


class MultiWorldFactory(WorldFactory):
    def create_maze_world(self, w, h):
        return MultiWorld(w, h)

    def klyubin_world_multi(self):
        maze = self.klyubin_world()
        maze.add_agent([5, 6], 'E')
        maze.add_agent([5, 1], 'N')
        return maze

class MultiWorld(MazeWorld):
    """ Represents an n x m grid world with walls at various locations and other agents.
        This agent controls the actions/states of other agents.
        The other agents' actions control the actions of this agent.
    """
    def __init__(self, height, width, toroidal = False):
        super().__init__(height, width, toroidal)
        self.agents = []

    def add_agent(self, position, a):
        position = np.array(position)
        self.agents.append(Agent(position, a))

    def in_collision(self, s, a, s_, a_):
        """ are or will be on same location or move through each other
        """
        state = self._index_to_cell(s)
        new_state = state + self.actions[a]
        if self._cell_to_index(state) == self._cell_to_index(s_):
            return True
        elif self._cell_to_index(new_state) == self._cell_to_index(s_ + self.actions[a_]):
            return True
        elif (s == self._cell_to_index(s_ + self.actions[a_])) and (self._cell_to_index(new_state) == self._cell_to_index(s_)):
            return True
        else:
            return False

    def act(self, s, a, s_, a_, prob=1.):
        """ get updated state after action
        s  : state, index of grid position
        a : action
        prob : probability of performing action
        """
        rnd = np.random.rand()
        if rnd > prob:
            a = np.random.choice(list(filter(lambda x: x != a, self.actions.keys())))
        state = self._index_to_cell(s)
        # hit wall
        if a not in self.adjacencies[state[0]][state[1]]:
            return s
        new_state = state + self.actions[a]
        # can't move off grid
        if self.toroidal:
            new_state = self.vecmod(new_state, self.dims)
        else:
            if np.any(new_state < np.zeros(2)) or np.any(new_state >= self.dims):
                return s
        if self.in_collision(s, a, s_, a_):
            return s
        return self._cell_to_index(new_state)

    def compute_model(self, det=1.):
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
            for agent in self.agents:
                for i, a in enumerate(self.actions.keys()):
                    s_new = self.act(s, a, agent.postion, agent.action)
                    s_unc = list(map(lambda x : self.act(s, x, agent.postion, agent.action), filter(lambda x : x != a, self.actions.keys())))
                    T[s_new, i, s] += det / len(self.agents)
                    for su in s_unc:
                        T[su, i, s] += ((1-det)/len(s_unc)) / len(self.agents)
        self.T = T
        return T


    def influence_on_other(self, det=1.):
        n_actions = len(self.actions)
        n_states = self.dims[0]*self.dims[1]
        # compute environment dynamics as a matrix T
        T = np.zeros([n_states, len(self.agents), n_actions,n_states])
        # T[s',a,s] is the probability of landing in s' given action a is taken in state s.
        for s in range(n_states):
            for k, agent in enumerate(self.agents):
                for i, a in enumerate(self.actions.keys()):
                    for j, a_k in enumerate(self.actions.keys()):
                            s_new = self.act(self._cell_to_index(agent.postion), a_k, self._index_to_cell(s), a)
                            T[s_new, k, j, s] += det
                            T[s_new, k, i, s] += det
        self.T = T.sum(axis=1)
        return T.sum(axis=1)


    def plot(self, fig, ax, pos=None, traj=None, action=None, colorMap=None, vmin=None, vmax=None):
        G = np.zeros(self.dims) if colorMap is None else colorMap.copy()
        # plot color map
        if vmax is not None:
            im = ax.pcolor(G, vmin=vmin, vmax=vmax)#, cmap = 'Greys')
        else:
            im = ax.pcolor(G)
        fig.colorbar(im, ax=ax)
        if pos is not None:
            ax.scatter([pos[1] + 0.5], [pos[0] + 0.5], s = 100, c = 'w')
        # plot trajectory
        if traj is not None:
            y, x = zip(*traj)
            y = np.array(y) + 0.5
            x = np.array(x) + 0.5
            ax.plot(x, y)
            ax.scatter([x[0]], [y[0]], s = 100, c = 'b')
            ax.scatter([x[-1]], [y[-1]], s = 100, c = 'r')
        for wall in self.walls:
            y, x = zip(*wall)
            (y, x) = ([max(y), max(y)], [x[0], x[0] + 1]) if x[0] == x[1] else ([y[0], y[0] + 1], [max(x), max(x)])
            ax.plot(x, y, c = 'w')
        for agent in self.agents:
            y, x = agent.postion
            ax.add_patch(patches.Circle((x+.5, y+.5), .5, linewidth=1, edgecolor='k', facecolor='w'))
            dir = self.actions[agent.action]
            ax.arrow(x+.5, y+.5, dir[1]/2, dir[0]/2)

        if action is not None:
            ax.set_title(str(action))


