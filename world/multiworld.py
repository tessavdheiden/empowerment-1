import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import pickle
import itertools


from agent import EmpMaxAgent
from world.mazeworld import MazeWorld, WorldFactory


class Agent(object):
    def __init__(self, T, det):
        self.brain = EmpMaxAgent(T=T, det=det)

    def set_s(self, s):
        self.s = s

    def set_cell(self, cell):
        self.cell = cell

    def set_action(self, action):
        self.action = action

    def set_a(self, a):
        self.a = a

    def decide(self, s):
        return self.brain.act(s)

    def rewire(self, s, a, s_):
        self.brain.update(s, a, s_)

    def get_cell(self):
        return self.cell

    def get_s(self):
        return self.s

    def get_action(self):
        return self.action

    def get_a(self):
        return self.a

    def predict_a(self, s):
        return self.brain.action_map[s]

    def predict_s(self, s, a):
        return np.argmax(self.brain.T[:, a, s])

    def save_params(self):
        params = dict(vars(self.brain))
        with open('params.pkl', 'wb') as f:
            pickle.dump(params, f)

    def load_params(self):
        with open('params.pkl', 'rb') as f:
            params = pickle.load(f)

        self.brain.T = params.get('T')
        self.brain.Q = params.get('Q')
        self.brain.R = params.get('R')
        self.brain.tau = params.get('tau')


class MultiWorld(MazeWorld):
    """ Represents an n x m grid world with walls at various locations and other agents.
        This agent controls the actions/states of other agents.
        The other agents' actions control the actions of this agent.
    """
    def __init__(self, height, width, toroidal = False):
        super().__init__(height, width, toroidal)
        self.agents = []
        self.locations = list()
        self.grids = list()
        self.a_list = list()
        self.n_s = height*width
        self.n_a = 0

    def add_agent(self, cell, action):
        cell = np.array(cell)
        emptymaze = MazeWorld(self.height, self.width)
        agent = Agent(T=emptymaze.compute_transition(), det=.9)
        agent.set_cell(cell)
        agent.set_s(self._cell_to_index(cell))
        agent.set_action(action)

        self.agents.append(agent)
        self.n_a = len(self.agents)

    def in_collision(self, s, a, s_, a_):
        """ are or will be on same location or move through each other
        behind wall or off-grid will prevent a collision
        """
        state = self._index_to_cell(s)
        new_state = state + self.actions[a]
        state_ = self._index_to_cell(s_)
        new_state_ = state_ + self.actions[a_]
        if self._cell_to_index(state) == s_: # can't be on same cell
            return True
        elif a_ not in self.adjacencies[state_[0]][state_[1]] or np.any(new_state_ < np.zeros(2)) or np.any(new_state_ >= self.dims): # the other cannot push if its behind a wall
           return False
        if self._cell_to_index(new_state) == self._cell_to_index(new_state_): # landing on same cell
            return True
        elif (s == self._cell_to_index(new_state_)) and (self._cell_to_index(new_state) == s_): # switching places
            return True
        else:
            return False

    def compute_ma_transition(self, n_a, det=1.):
        self.locations = np.array(list(itertools.permutations(np.arange(self.n_s), n_a)))
        self.a_list = list(itertools.product(self.actions.keys(), repeat=n_a))
        n_actions = len(self.a_list)

        n_configs = len(self.locations)
        # compute environment dynamics as a matrix T
        T = np.zeros([n_configs, n_actions, n_configs])
        # T[s',a,s] is the probability of landing in s' given action a is taken in state s.

        for c, locs in enumerate(self.locations):
            for i, alist in enumerate(self.a_list):
                locs_new = [self.act(locs[j], alist[j], det) for j in range(n_a)]

                # any of the agents on same location, do not move
                if len(set(locs_new)) < n_a or (locs[0] == locs_new[1] and locs[1] == locs_new[0]):
                    locs_new = locs

                c_new = self._location_to_index(locs_new)
                T[c_new, i, c] += det

                if det == 1: continue
                locs_unc = np.array([list(map(lambda x: self.act(locs[j], x, det), filter(lambda x: x != alist[j], self.actions.keys()))) for j in range(n_a)]).T
                assert locs_unc.shape == ((len(self.actions) - 1), n_a)

                for lu in locs_unc:
                    if np.all(lu == lu[0]): continue # collision
                    c_unc = self._location_to_index(lu)
                    T[c_unc, i, c] += (1 - det) / (len(locs_unc))

        self.T = T
        return T

    def interact(self, det=1.):
        """ Computes new locations and update actions for each agent if no collisions.
        """
        new_s = list()
        old_s = [agent.s for agent in self.agents]
        assert len(set(old_s)) == self.n_a

        for i, agent in enumerate(self.agents):
            a = agent.decide(old_s[i])
            action = list(self.actions.keys())[a]
            new_s.append(self.act(agent.s, action))
            agent.set_a(a)
            agent.set_action(action)

        # collision
        if len(set(new_s)) < self.n_a:
            new_s = old_s

        # update q-function
        for i, agent in enumerate(self.agents):
            agent.rewire(old_s[i], agent.get_a(), new_s[i])
            s_ = new_s[i]
            agent.set_s(s_)
            agent.set_cell(self._index_to_cell(s_))

        return new_s

    def predict_trajectory(self, agent, target, t_step):
        """ Computes new locations of one agent (target), based on the value function of another (agent).
        """
        s = self.act(target.get_s(), target.get_action())
        traj = [self._index_to_cell(s)]

        for t in range(t_step):
            a = agent.predict_a(s)
            s = agent.predict_s(s, a)#self.act(s, list(self.actions.keys())[a])
            traj.append(self._index_to_cell(s))

        return traj

    def _index_to_location(self, c, j):
        return self.locations[c][j]

    def _location_to_index(self, locs):
        return np.where(np.all(self.locations == locs, axis=1))[0][0]

    def plot(self, fig, ax, pos=None, traj=None, action=None, colorMap=None, vmin=None, vmax=None, cmap='viridis'):
        ax.clear()
        G = np.zeros(self.dims) if colorMap is None else colorMap.copy()
        # plot color map
        if vmax is not None:
            im = ax.pcolor(G, vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            im = ax.pcolor(G, cmap=cmap)
        #fig.colorbar(im, ax=ax)
        if pos is not None:
            ax.scatter([pos[1] + 0.5], [pos[0] + 0.5], s = 100, c = 'w')
        # plot trajectory
        if traj is not None:
            y, x = zip(*traj)
            y = np.array(y) + 0.5
            x = np.array(x) + 0.5
            ax.plot(x, y, c = 'b')

        if action is not None:
            ax.set_title(str(action))

    def plot_entities(self, fig, ax):
        for wall in self.walls:
            y, x = zip(*wall)
            (y, x) = ([max(y), max(y)], [x[0], x[0] + 1]) if x[0] == x[1] else ([y[0], y[0] + 1], [max(x), max(x)])
            ax.plot(x, y, c = 'w')

        for i, agent in enumerate(self.agents):
            y, x = self._index_to_cell(agent.s)
            ax.add_patch(patches.Circle((x+.5, y+.5), .5, linewidth=1, edgecolor='k', facecolor='w'))
            dir = self.actions[agent.get_action()]
            ax.arrow(x+.5, y+.5, dir[1]/2, dir[0]/2)
            ax.text(x+.5, y+.25, i, horizontalalignment='center', verticalalignment='center')


class MultiWorldFactory(WorldFactory):
    def create_maze_world(self, height, width):
        return MultiWorld(height, width)

    def klyubin_2agents(self):
        maze = self.klyubin_world()
        maze.add_wall([4, 4], 'E')
        maze.add_wall([5, 4], 'E')
        maze.add_agent([1, 3], '_')
        maze.add_agent([4, 3], 'E')
        return maze

    def door_2agents(self):
        maze = self.door_world()
        maze.add_agent([1, 3], '_')
        maze.add_agent([4, 3], 'E')
        return maze

    def door2_2agents(self):
        maze = self.door2_world()
        maze.add_agent([1, 3], '_')
        maze.add_agent([4, 3], 'E')
        return maze

    def tunnel_2agents(self):
        maze = self.tunnel_world()
        maze.add_agent([1, 3], '_')
        maze.add_agent([4, 3], 'E')
        return maze

    def step_3agents(self):
        maze = self.step_world()
        maze.add_agent([1, 3], '_')
        maze.add_agent([4, 3], 'E')
        maze.add_agent([4, 1], 'E')
        return maze

    def simple_2agents(self):
        maze = self.simple()
        maze.add_agent([0, 1], '_')
        maze.add_agent([1, 0], 'E')
        return maze


