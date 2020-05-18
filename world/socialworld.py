import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import itertools

from world.multiworld import MultiWorld, Agent


class SocialWorld(MultiWorld):
    """ Represents an n x m grid world with walls at various locations and other agents.
        This agent controls the actions/states of other agents.
        The other agents' actions control the actions of this agent.
    """
    def __init__(self, height, width, toroidal = False):
        super().__init__(height, width, toroidal)


    def add_social_agent(self, position, action, T):
        position = np.array(position)
        agent = Agent(T=T.copy(), det=.9, n_step=1)
        agent.set(position, action, self._cell_to_index(position), self.dims)
        self.agents.append(agent)
        self.n_a = len(self.agents)

    def interact(self, n_a, det=1.):
        """ Computes probabilistic model T[s',a,s] corresponding to the maze world.
        det : float between 0 and 1
            Probability of action successfully performed (otherwise a random different action is performed with probability 1 - det). When det = 1 the dynamics are deterministic.
        """
        locs = [agent.s for agent in self.agents]
        c = self._location_to_index(locs)
        locs_new = []
        a_new = []
        for i, agent in enumerate(self.agents):
            a = agent.act(c)
            a = np.random.randint(len(self.a_list))

            action = self.a_list[a][i]
            a_new.append(a)

            agent.action = action
            s_ = self.act(locs[i], action)
            locs_new.append(s_)

        # collision
        if len(set(locs_new)) != n_a:
            locs_new = locs

        c_new = self._location_to_index(locs_new)[0][0]
        for i, agent in enumerate(self.agents):
            s_ = locs_new[i]
            pos = self._index_to_cell(s_)
            agent.visited[pos[0], pos[1]] += 1
            agent.position = pos
            agent.s = s_
            agent.update(c, a_new[i], c_new)

from world.mazeworld import WorldFactory

class SocialWorldFactory(WorldFactory):
    def create_maze_world(self, height, width):
        return SocialWorld(height, width)

    def simple_2agents(self):
        maze = self.simple()
        T = maze.compute_ma_transition(2, det=.1)
        maze.add_social_agent([1, 0], 'E', T)
        maze.add_social_agent([0, 1], 'E', T)
        return maze

    def door2_2agents(self):
        maze = self.door2_world()
        T = maze.compute_ma_transition(2, det=.1)
        maze.add_social_agent([1, 3], '_', T)
        maze.add_social_agent([4, 3], 'E', T)
        return maze

    def klyubin_world_2agents(self):
        maze = self.klyubin_world()
        T = maze.compute_ma_transition(2, det=1.)
        maze.add_social_agent([1, 3], '_', T)
        maze.add_social_agent([4, 3], 'E', T)
        return maze

