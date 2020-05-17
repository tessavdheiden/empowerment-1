import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import itertools


from agent import EmpMaxAgent
from world.mazeworld import MazeWorld, WorldFactory


class Agent(EmpMaxAgent):
    def __init__(self, T, det):
        super().__init__(T, det, alpha=0.1, gamma=0.9, n_step=2, n_samples=1000)

    def set(self, position, action, s, dims):
        self.s = s
        self.s_ = s
        self.position = position
        self.action = action
        self.visited = np.zeros(dims)


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

    def add_agent(self, position, action):
        position = np.array(position)
        emptymaze = MazeWorld(self.height, self.width)
        agent = Agent(T=emptymaze.compute_transition(), det=1.)
        agent.set(position, action, self._cell_to_index(position), self.dims)
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
        self.locations = list(itertools.combinations(np.arange(self.n_s), n_a))
        self.grids = np.zeros((len(self.locations), self.n_s), dtype="int8")
        self.grids[np.arange(len(self.locations))[None].T, self.locations] = 1
        self.a_list = list(itertools.product(self.actions.keys(), repeat=n_a))
        n_actions = len(self.a_list)

        n_configs = len(self.grids)
        # compute environment dynamics as a matrix T
        T = np.zeros([n_configs, n_actions, n_configs])
        # T[s',a,s] is the probability of landing in s' given action a is taken in state s.

        for c, locs in enumerate(self.locations):
            for i, alist in enumerate(self.a_list):
                locs_new = [self.act(locs[j], alist[j]) for j in range(n_a)]

                # any of the agents on same location, do not move
                if len(set(locs_new)) != n_a:
                    locs_new = locs

                grid = self._location_to_grid(locs_new)
                c_new = self._grid_to_index(grid)
                T[c_new, i, c] += det

        self.T = T
        return T

    def _index_to_location(self, c, j):
        return self.locations[c][j]

    def _location_to_grid(self, locs):
        grid = np.zeros(self.n_s)
        for l in locs:
            grid[l] = 1
        return grid

    def _location_to_index(self, locs):
        grid = self._location_to_grid(locs)
        return self._grid_to_index(grid)

    def _grid_to_index(self, grid):
        return np.where(np.all(self.grids == grid, axis=1))

    def interact(self, det=1.):
        """ Computes probabilistic model T[s',a,s] corresponding to the maze world.
        det : float between 0 and 1
            Probability of action successfully performed (otherwise a random different action is performed with probability 1 - det). When det = 1 the dynamics are deterministic.
        """
        for i, agent in enumerate(self.agents):
            s = agent.s
            a = agent.act(s)
            agent.action = list(self.actions.keys())[a]
            s_ = self.act(s, agent.action)

            for other in self.agents[:i] + self.agents[i+1:]:
                if self.in_collision(s_, agent.action, other.s, other.action):
                    s_ = s
                    break

            pos = self._index_to_cell(s_)
            agent.visited[pos[0], pos[1]] += 1

            agent.update(s, a, s_)
            agent.s = s_
            agent.position = pos


    def predict(self, agent, s, a, n_step):
        traj = np.zeros((n_step+1, 2))
        traj[0, :] = self._index_to_cell(s)
        new_state = self._index_to_cell(s) + self.actions[a]
        state = self._index_to_cell(s) if np.any(new_state < np.zeros(2)) or np.any(new_state >= self.dims) else new_state
        for t in range(n_step):
            traj[t+1, :] = state
            a = agent.action_map[s]
            new_state = state + list(self.actions.values())[a]
            state = state if np.any(new_state < np.zeros(2)) or np.any(new_state >= self.dims) else new_state
            s = self._cell_to_index(state)

        return traj

    def plot(self, fig, ax, pos=None, traj=None, action=None, colorMap=None, vmin=None, vmax=None,cmap='viridis'):
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
            ax.plot(x, y, c = 'k')
        for wall in self.walls:
            y, x = zip(*wall)
            (y, x) = ([max(y), max(y)], [x[0], x[0] + 1]) if x[0] == x[1] else ([y[0], y[0] + 1], [max(x), max(x)])
            ax.plot(x, y, c = 'w')

        for i, agent in enumerate(self.agents):
            agent.position = self._index_to_cell(agent.s)
            y, x = agent.position
            ax.add_patch(patches.Circle((x+.5, y+.5), .5, linewidth=1, edgecolor='k', facecolor='w'))
            dir = self.actions[agent.action]
            ax.arrow(x+.5, y+.5, dir[1]/2, dir[0]/2)
            ax.text(x+.5, y+.25, i, horizontalalignment='center', verticalalignment='center')

        if action is not None:
            ax.set_title(str(action))



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


