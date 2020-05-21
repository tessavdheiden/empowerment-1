import numpy as np
import copy

from world.multiworld import MultiWorld, Agent


class SocialWorld(MultiWorld):
    """ Represents an n x m grid world with walls at various locations and other agents.
        This agent controls the actions/states of other agents.
        The other agents' actions control the actions of this agent.
    """
    def __init__(self, height, width, toroidal = False):
        super().__init__(height, width, toroidal)

    def add_social_agent(self, cell, action, T):
        cell = np.array(cell)
        T = copy.deepcopy(T)
        agent = Agent(T=T, det=1)
        agent.set_cell(cell)
        agent.set_s(self._cell_to_index(cell))
        agent.set_action(action)

        self.agents.append(agent)
        self.n_a = len(self.agents)


from world.mazeworld import WorldFactory

class SocialWorldFactory(WorldFactory):
    def create_maze_world(self, height, width):
        return SocialWorld(height, width)

    def simple_2agents(self):
        maze = self.simple()
        emptymaze = self.create_maze_world(height=maze.height, width=maze.width)
        T = emptymaze.compute_ma_transition(2, det=1)
        maze.add_social_agent([0, 1], '_', T)
        maze.add_social_agent([1, 0], 'E', T)
        return maze

    def door2_2agents(self):
        maze = self.door2_world()
        emptymaze = self.create_maze_world(height=maze.height, width=maze.width)
        T = emptymaze.compute_ma_transition(2, det=1)
        maze.add_social_agent([1, 3], '_', T)
        maze.add_social_agent([4, 3], 'E', T)
        return maze

    def klyubin_world_2agents(self):
        maze = self.klyubin_world()
        emptymaze = self.create_maze_world(height=maze.height, width=maze.width)
        T = emptymaze.compute_ma_transition(2, det=1)
        maze.add_social_agent([1, 3], '_', T)
        maze.add_social_agent([4, 3], 'E', T)
        return maze

