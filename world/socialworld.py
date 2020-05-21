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

    def interact_ma(self, n_a, det=1.):
        """ Computes new locations and update actions for each agent if no collisions.
        """
        c = np.random.randint(len(self.locations))

        i = self.agents[0].decide(c)
        #i = np.random.randint(len(self.a_list))

        locs = [agent.s for agent in self.agents]
        locs_new = [self.act(self.locations[c][j], self.a_list[i][j], det) for j in range(n_a)]

        # any of the agents on same location, do not move
        if len(set(locs_new)) < n_a:
            locs_new = locs

        assert len(locs_new) == n_a
        c_new = self._location_to_index(locs_new)[0] #np.argmax(self.T[:, i, c])
        for j, agent in enumerate(self.agents):
            agent.rewire(c, i, c_new)
            agent.brain.update_tau()
            agent.set_s(self._index_to_location(c_new, j))
            agent.set_action(self.a_list[i][j])
            agent.set_cell(self._index_to_cell(agent.s))

        #
        # new_s = list()
        # old_s = [agent.s for agent in self.agents]
        # old_c = self._location_to_index(old_s)
        # for i, agent in enumerate(self.agents):
        #     a = agent.decide(old_c)
        #     a = np.random.randint(len(self.a_list))
        #     a_lst = self.a_list[a]
        #     new_s.append([self.act(agent.s, action) for action in a_lst]) # agent i chooses actions for other agents
        #     agent.set_a(a)
        #     agent.set_action(a_lst[i])
        #
        # for i, agent in enumerate(self.agents):
        #     # collision
        #     if len(set(new_s[i])) < self.n_a: # TODO: switch places allowed
        #         new_s[i] = old_s
        #
        # assert (len(new_s) == n_a) and (len(new_s[0]) == n_a)
        # # update q-function
        # for i, agent in enumerate(self.agents):
        #     new_c = self._location_to_index(new_s[0])
        #     a = self.agents[0].get_a()
        #     agent.rewire(old_c.item(0), a, new_c.item(0))
        #     s_ = new_s[0][i]
        #     agent.set_s(s_)
        #     agent.set_cell(self._index_to_cell(s_))
        #     agent.set_a(a)
        #     agent.set_action(self.agents[0].get_action())

from world.mazeworld import WorldFactory

class SocialWorldFactory(WorldFactory):
    def create_maze_world(self, height, width):
        return SocialWorld(height, width)

    def empty_2agents(self):
        maze = self.create_maze_world(height=6, width=3)
        T = maze.compute_ma_transition(2, det=1)
        maze.add_social_agent([0, 1], '_', T)
        maze.add_social_agent([1, 0], 'E', T)
        return maze

    def simple_2agents(self):
        maze = self.simple()
        emptymaze = self.create_maze_world(height=maze.height, width=maze.width)
        T = emptymaze.compute_ma_transition(2, det=1)
        maze.add_social_agent([0, 1], '_', T)
        maze.add_social_agent([1, 0], 'E', T)
        return maze

    def door2_2agents(self):
        maze = self.door2_world()
        T = maze.compute_ma_transition(2, det=1)
        maze.add_social_agent([1, 3], '_', T)
        maze.add_social_agent([4, 3], 'E', T)
        return maze

    def klyubin_world_2agents(self):
        maze = self.klyubin_world()
        T = maze.compute_ma_transition(2, det=1)
        maze.add_social_agent([1, 3], '_', T)
        maze.add_social_agent([4, 3], 'E', T)
        return maze

