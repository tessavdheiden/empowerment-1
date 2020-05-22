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

    def interact(self, c, eval=False, det=1.):
        """ Computes new configuration if no collisions.
        """
        s = [agent.s for agent in self.agents]
        a_lst = [agent.decide(c, eval==False) for agent in self.agents]  # [0, 1]
        action_lst = [self.a_list[a][i] for i, a in enumerate(a_lst)]  # [('N', 'N'), ('N', 'S')] --> ['N','S']
        a = np.argmax(np.array([x[0] == action_lst[0] and x[1] == action_lst[1] for x in self.a_list]))  # [1]
        s_ = [self.act(agent.s, action) for agent, action in zip(self.agents, action_lst)]

        # collision
        if len(set(s_)) < self.n_a or (s[0] == s_[1] and s[1] == s_[0]):
            s_ = [self._index_to_location(c, i) for i in range(self.n_a)]

        c_ = self._location_to_index(s_)
        for i, agent in enumerate(self.agents):
            agent.set_s(s_[i])
            agent.set_a(a_lst[i])
            agent.set_action(action_lst[i])

        # update Q functions
        for i, agent in enumerate(self.agents):
            agent.rewire(c, a, c_)

        return c_

    def predict_trajectory(self, agent, t_step):
        """ Computes new locations of one agent (target), based on the value function of another (agent).
        """
        locs = [agent.s for agent in self.agents]
        c = self._location_to_index(locs)
        traj = np.zeros((t_step+1, self.n_a, 2))
        traj[0:1, :, :] = np.asarray([self._index_to_cell(s) for s in locs])

        for t in range(t_step):
            a = agent.predict_a(c)
            c = agent.predict_s(c, a)#self.act(s, list(self.actions.keys())[a])
            locs = [self._index_to_location(c, i) for i in range(self.n_a)]
            traj[t+1, :, :] = np.asarray([self._index_to_cell(s) for s in locs])
        traj = np.asarray(traj).reshape(t_step+1, self.n_a, 2)
        return traj


from world.mazeworld import WorldFactory

class SocialWorldFactory(WorldFactory):
    def create_maze_world(self, height, width):
        return SocialWorld(height, width)

    def simple_2agents(self):
        maze = self.simple()
        emptymaze = self.create_maze_world(height=maze.height, width=maze.width)
        T = emptymaze.compute_ma_transition(2, det=1)
        maze.add_social_agent([0, 1], '_', T)
        maze.add_social_agent([0, 0], 'E', T)
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

