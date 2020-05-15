import numpy as np
from world.world import World

class Pendulum(World):
    """ Represents an n x m grid world with walls at various locations. Actions can be performed (N, S, E, W, "stay") moving a player around the grid world. You can't move through walls. """
    g = 10.0
    l = 1.0
    m = 1.0
    dt = 0.05

    def __init__(self, height, width, toroidal = False):
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
        it = np.linspace(-10, 10, 5)
        self.actions = dict(zip(it, it))

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
        new_state = self.vecmod(new_state, self.dims)
        return self._cell_to_index(new_state)

    def plot(self, fig, ax, pos = None, traj = None, action = None, colorMap = None, vmin = None, vmax = None):
        G = np.zeros(self.dims) if colorMap is None else colorMap.copy()
        # plot color map
        if vmax is not None:
            im = ax.pcolor(G, vmin = vmin, vmax=vmax) # , cmap = 'Greys')
        else:
            im = ax.pcolor(G)
        fig.colorbar(im, ax=ax)
        if pos is not None:
            ax.scatter([pos[1] + 0.5], [pos[0] + 0.5], s = 100, c = 'w')
        # plot trajectory
        if traj is not None:
            y, x = zip(*traj)
            y = np.array(y) + 0.5
            x = np.array(x) + 0.5
            ax.plot(x, y)
            ax.scatter([x[0]], [y[0]], s = 100, c = 'b')
            ax.scatter([x[-1]], [y[-1]], s = 100, c = 'r')

        if action is not None:
            ax.set_title(str(action))

        ax.set_xticks(np.linspace(0, len(self.angles), 3))
        ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        ax.set_xlabel('angle')
        ax.set_yticks(np.linspace(0, len(self.velocities), 3))
        ax.set_yticklabels(np.round(np.linspace(self.velocities[0], self.velocities[-1], 3)))

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx