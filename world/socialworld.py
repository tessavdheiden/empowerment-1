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
        agent = Agent(T=T.copy(), det=1.)
        agent.set(position, action, self._cell_to_index(position), self.dims)
        self.agents.append(agent)
        self.n_a = len(self.agents)

    def interact(self, n_a, det=1.):
        """ Computes probabilistic model T[s',a,s] corresponding to the maze world.
        det : float between 0 and 1
            Probability of action successfully performed (otherwise a random different action is performed with probability 1 - det). When det = 1 the dynamics are deterministic.
        """
        a_list = list(itertools.product(self.actions.keys(), repeat=n_a))
        for i, agent in enumerate(self.agents):
            s = agent.s
            a = agent.act(s)
            action = a_list[a][i]
            agent.action = action
            s_ = self.act(s, action)

            for other in self.agents[:i] + self.agents[i+1:]:
                if self.in_collision(s_, action, other.s, other.action):
                    s_ = s

            pos = self._index_to_cell(s_)
            agent.visited[pos[0], pos[1]] += 1

            agent.update(s, a, s_)
            agent.s = s_
            agent.position = pos

from world.mazeworld import WorldFactory

class SocialWorldFactory(WorldFactory):
    def create_maze_world(self, height, width):
        return SocialWorld(height, width)

    def simple_2agents(self):
        maze = self.klyubin_world()
        T = maze.compute_ma_transition(2)
        maze.add_social_agent([2, 0], 'E', T)
        maze.add_social_agent([2, 1], 'E', T)
        return maze

