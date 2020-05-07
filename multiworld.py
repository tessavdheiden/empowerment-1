import numpy as np


class Agent(object):
    def __init__(self, state, action):
        self.state = state
        self.action = action


from mazeworld import MazeWorld, WorldFactory


class MultiWorldFactory(WorldFactory):
    def create_maze_world(self, w, h):
        return MultiWorld(w, h)

    def klyubin_world_multi(self):
        maze = self.klyubin_world()
        centre = np.array([maze.width, maze.height], dtype=int) // 2
        maze.add_agent(centre, '_')
        return maze

class MultiWorld(MazeWorld):
    """ Represents an n x m grid world with walls at various locations and other agents.
        This agent controls the actions/states of other agents.
        The other agents' actions control the actions of this agent.
    """
    def __init__(self, height, width, toroidal = False):
        MazeWorld.__init__(self, height, width, toroidal)
        self.agents = []

    def make_cell_occupied(self, cell):
        self.adjacencies[cell[0]][cell[1]] = list()

    def add_agent(self, position, a):
        position = np.array(position)
        if a == '_':
            self.make_cell_occupied(position)
        elif a == 'N' and position[0] < self.height:
            self.make_cell_occupied(position + self.actions[a])
        elif a == 'S' and position[0] > 0:
            self.make_cell_occupied(position - self.actions[a])
        elif a == 'E' and position[1] < self.width:
            self.make_cell_occupied(position + self.actions[a])
        elif a == 'W' and position[1] > 0:
            self.make_cell_occupied(position - self.actions[a])
        self.agents.append(Agent(position, a))

    def influence_on_others(self, det=1.):
        """ Computes probabilistic model T[s',a,s] corresponding to the maze world.
        det : float between 0 and 1
            Probability of action successfully performed (otherwise a random different action is performed with probability 1 - det). When det = 1 the dynamics are deterministic.
        """
        n_actions = len(self.actions)
        n_states = self.dims[0]*self.dims[1]
        # compute environment dynamics as a matrix T
        T = np.zeros([n_states, i, n_actions,n_states])
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

    def incluence_on_me(self):
        pass


