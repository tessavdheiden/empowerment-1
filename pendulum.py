import numpy as np
from mazeworld import MazeWorld

class Pendulum(MazeWorld):
    """ Represents an n x m grid world with walls at various locations. Actions can be performed (N, S, E, W, "stay") moving a player around the grid world. You can't move through walls. """
    g = 10.0
    l = 1.0
    m = 2.0
    dt = 0.05

    def __init__(self, height, width, toroidal = True):
        """
        height : int
            Height of grid world
        width : int
            Width of grid world
        toroidal:
            If true, player can move off the edge of the world, appearing on the other side.
        """
        self.dims = [height, width]
        self.height = height
        self.width = width
        self.adjacencies = dict()
        self.actions = {
            "FR" : np.array([20]),  # FAST RIGHT
            "R": np.array([10]),    # FAST RIGHT
            "SR" : np.array([5]),   # SLOW RIGHT
            "_" : np.array([0]),    # STAY
            "SL" : np.array([-5]),  # SLOW RIGHT
            "L": np.array([-10]),   # FAST LEFT
            "FL" : np.array([-20])  # FAST LEFT
        }

        for i in range(height):
            self.adjacencies[i] = dict()
            for j in range(width):
                self.adjacencies[i][j] = list(self.actions.keys())
        self.walls = []
        self.toroidal = toroidal
        self.vecmod = np.vectorize(lambda x, y : x % y)
        self.angles = np.linspace(-np.pi, np.pi, width)
        self.velocities = np.linspace(-8, 8, height)

    def discretize(self, pos, vel):
        return np.array([find_nearest(self.velocities, vel), find_nearest(self.angles, pos)])

    def forward(self, s, a):
        th = self.angles[s[1]]
        thdot = self.velocities[s[0]]

        newthdot = thdot + (-3 * self.g / (2 * self.l) * np.sin(th + np.pi) + 3. / (self.m * self.l ** 2) * a) * self.dt
        newth = th + newthdot * self.dt

        newthdot = np.clip(newthdot, -8, 8)

        newth = angle_normalize(newth)

        return self.discretize(newth, newthdot)

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
        if a not in self.adjacencies[state[0]][state[1]]:
            return self._cell_to_index(state)
        new_state = self.forward(state, self.actions[a])
        # can't move off grid
        if self.toroidal:
            new_state = self.vecmod(new_state, self.dims)
        else:
            if np.any(new_state < np.zeros(2)) or np.any(new_state >= self.dims):
                return self._cell_to_index(state)
        return self._cell_to_index(new_state)

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx