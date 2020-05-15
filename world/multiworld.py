import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import itertools


from agent import EmpMaxAgent


class Agent(EmpMaxAgent):
    def __init__(self, T, det):
        super().__init__(T, det, alpha=0.1, gamma=0.9, n_step=2, n_samples=1000)

    def set(self, position, action, s, dims):
        self.s = s
        self.s_ = s
        self.position = position
        self.action = action
        self.visited = np.zeros(dims)


from world.mazeworld import MazeWorld, WorldFactory


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

    def door_3agents(self):
        maze = self.door_world()
        maze.add_agent([1, 3], '_')
        maze.add_agent([4, 3], 'E')
        maze.add_agent([5, 3], 'E')
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
        maze.add_agent([1, 3], '_')
        maze.add_agent([4, 3], 'E')
        return maze



class MultiWorld(MazeWorld):
    """ Represents an n x m grid world with walls at various locations and other agents.
        This agent controls the actions/states of other agents.
        The other agents' actions control the actions of this agent.
    """
    def __init__(self, height, width, toroidal = False):
        super().__init__(height, width, toroidal)
        self.agents = []

    def add_agent(self, position, action):
        position = np.array(position)
        emptymaze = MazeWorld(self.height, self.width)
        agent = Agent(T=emptymaze.compute_model(), det=1.)
        agent.set(position, action, self._cell_to_index(position), self.dims)
        self.agents.append(agent)

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

    def act(self, s, a, prob=1.):
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

        return self._cell_to_index(new_state)

    def compute_model(self, det=1.):
        """ Computes probabilistic model T[s',a,s] corresponding to the maze world.
        det : float between 0 and 1
            Probability of action successfully performed (otherwise a random different action is performed with probability 1 - det). When det = 1 the dynamics are deterministic.
        """
        n_actions = len(self.actions)
        n_states = self.dims[0]*self.dims[1]
        # compute environment dynamics as a matrix T
        T = np.zeros([n_states, n_actions, n_states])
        # T[s',a,s] is the probability of landing in s' given action a is taken in state s.
        for s in range(n_states):
            for i, a in enumerate(self.actions.keys()):
                s_new = self.act(s, a)
                s_unc = list(map(lambda x : self.act(s, x), filter(lambda x : x != a, self.actions.keys())))
                T[s_new, i, s] += det
                for su in s_unc:
                    T[su, i, s] += ((1-det)/len(s_unc))
        self.T = T
        return T

    def generate_configs(self, n_states, n_agents):
        which = np.array(list(itertools.combinations(range(n_states), n_agents)))
        grid = np.zeros((len(which), n_states), dtype="int8")
        grid[np.arange(len(which))[None].T, which] = 1
        return grid

    def _omap_to_index(self, omap):
        return np.where(np.all(self.configs == omap, axis=1))

    def compute_ma_model(self, det=1.):
        def nCr(n, r):
            f = math.factorial
            return int(f(n) / f(n - r) / f(r))

        n_agents = len(self.agents)
        n_states = self.dims[0]*self.dims[1]
        n_configs = nCr(n_states, n_agents)
        alists = [l for l in itertools.product(self.actions.keys(), repeat=n_agents)]
        self.slists = list(itertools.combinations(np.arange(n_states), n_agents))
        n_actions = len(alists)

        self.configs = self.generate_configs(n_states, n_agents)
        self.slists_new = np.zeros((n_configs, n_actions, n_agents))

        # compute environment dynamics as a matrix T
        T = np.zeros([n_configs, n_actions, n_configs])
        # T[s',a,s] is the probability of landing in s' given action a is taken in state s.

        for c in range(n_configs):
            states = self.slists[c]
            for i, alist in enumerate(alists):
                for j, agent in enumerate(self.agents):
                    agent.s = states[j]
                    #agent.position = self._index_to_cell(states[j])
                    agent.action = alist[j]
                # self.plot(fig, ax[0,0], colorMap=np.zeros(self.dims))
                # ax[0,0].set_title(f"state={c}")

                seen = set()
                for agent in self.agents:
                    s_ = self.act(agent.s, agent.action)
                    seen.add(s_)

                # any of the agents on same location, do not move
                n_hot = np.zeros(n_states)
                if len(seen) == n_agents:
                    for agent in self.agents:
                        agent.s = self.act(agent.s, agent.action)
                        #agent.position = self._index_to_cell(agent.s)

                # self.plot(fig, ax[0, 1], colorMap=np.zeros(self.dims))
                # ax[0, 1].set_title(f"in collision ={len(seen) != n_agents}")
                # plt.pause(.001)

                for j, agent in enumerate(self.agents):
                    n_hot[agent.s] = 1
                    self.slists_new[c][i][j] = agent.s

                assert sum(n_hot) == n_agents

                c_new = np.where(np.all(self.configs == n_hot, axis=1))
                T[c_new, i, c] += det

        self.T = T
        return T


    def interact(self, det=1.):
        """ Computes probabilistic model T[s',a,s] corresponding to the maze world.
        det : float between 0 and 1
            Probability of action successfully performed (otherwise a random different action is performed with probability 1 - det). When det = 1 the dynamics are deterministic.
        """
        for i, agent in enumerate(self.agents):
            s = agent.s
            a = agent.act(s)
            action = list(self.actions.keys())[a]
            agent.action = action
            s_ = self.act(s, action)

            influence = 0
            for j, other in enumerate(self.agents[:i] + self.agents[i+1:]):
                s_unc = set(map(lambda x : self.act(other.s, x), self.actions.keys())) # TODO: how many are already in collision of other agent
                if self.in_collision(s_, action, other.s, other.action):
                    influence -= 1 / len(s_unc)
                    s_ = s

            pos = self._index_to_cell(s_)
            agent.visited[pos[0], pos[1]] += 1

            agent.update(s, a, s_, influence)
            agent.s = s_
            agent.position = pos

    def predict(self, action_map, s, a, n_step):
        traj = np.zeros((n_step+1, 2))
        traj[0, :] = self._index_to_cell(s)
        new_state = self._index_to_cell(s) + self.actions[a]
        state = self._index_to_cell(s) if np.any(new_state < np.zeros(2)) or np.any(new_state >= self.dims) else new_state
        for t in range(n_step):
            traj[t+1, :] = state
            a = action_map[s]
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




