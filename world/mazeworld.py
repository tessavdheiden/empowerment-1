import numpy as np
import matplotlib.pyplot as plt

from world.world import World

class MazeWorld(World):
    """ Represents an n x m grid world with walls at various locations. Actions can be performed (N, S, E, W, "stay") moving a player around the grid world. You can't move through walls. """

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
        self.actions = {
            "N" : np.array([1, 0]), # UP
            "S" : np.array([-1,0]),  # DOWN
            "E" : np.array([0, 1]), # RIGHT
            "W" : np.array([0,-1]), # LEFT
            "_" : np.array([0, 0])  # STAY
        }
        self.opposite = {
            "N" : "S",
            "S" : "N",
            "W" : "E",
            "E" : "W"
        }
        for i in range(height):
            self.adjacencies[i] = dict()
            for j in range(width):
                self.adjacencies[i][j] = list(self.actions.keys())
        self.walls = []
        self.toroidal = toroidal
        self.vecmod = np.vectorize(lambda x, y : x % y)

    def add_wall(self, cell, direction):
        cell = np.array(cell)
        # remove action 
        self.adjacencies[cell[0]][cell[1]].remove(direction)
        # remove opposite action
        new_cell = cell + self.actions[direction]
        self.adjacencies[new_cell[0]][new_cell[1]].remove(self.opposite[direction])
        # save wall for plotting 
        self.walls.append((cell, new_cell))
        self.T = None

    def act(self, s, a, prob = 1.):
        """ get updated state after action
    
        s  : state, index of grid position 
        a : action 
        prob : probability of performing action
        """
        rnd = np.random.rand()
        if rnd > prob:
            a = np.random.choice(list(filter(lambda x : x !=a, self.actions.keys())))
        state = self._index_to_cell(s)
        if a not in self.adjacencies[state[0]][state[1]]:
            return self._cell_to_index(state)
        new_state = state + self.actions[a] 
        # can't move off grid
        if self.toroidal:
            new_state = self.vecmod(new_state, self.dims)
        else:
            if np.any(new_state < np.zeros(2)) or np.any(new_state >= self.dims):
                return self._cell_to_index(state)
        return self._cell_to_index(new_state)
    
    def act_nstep(self, s, actions):
        for a in actions:
            s = self.act(s, a)
        return s
    
    def plot(self, fig, ax, pos=None, traj=None, action=None, colorMap=None, vmin=None, vmax=None):
        G = np.zeros(self.dims) if colorMap is None else colorMap.copy()
        # plot color map
        if vmax is not None:
            im = plt.pcolor(G, vmin=vmin, vmax=vmax) # , cmap = 'Greys')
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
        for wall in self.walls:
            y, x = zip(*wall)
            (y, x) = ([max(y), max(y)], [x[0], x[0] + 1]) if x[0] == x[1] else ([y[0], y[0] + 1], [max(x), max(x)])
            ax.plot(x, y, c = 'w')
        if action is not None:
            ax.set_title(str(action))

    def print_state_numbers(self, fig, ax):
        nums = np.array([[i * self.dims[1] + j for j in range(self.dims[1])] for i in range(self.dims[0])])
        im = ax.pcolor(nums*0)
        fig.colorbar(im, ax=ax)
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                ax.text(j + .5, i + .5, nums[i, j],ha="center", va="center", color="w")


class WorldFactory(object):
    def create_maze_world(self, height, width):
        return MazeWorld(height, width)

    def klyubin_world(self):
        """ Build mazeworld from Klyubin et al. """
        maze = self.create_maze_world(10,10)
        # wall A
        for i in range(6):
            maze.add_wall( (1, i), "N" )
        # wall B & D
        for i in range(2):
            maze.add_wall( (i+2, 5), "E")
            maze.add_wall( (i+2, 6), "E")
        # wall C
        maze.add_wall( (3, 6), "N")
        # wall E
        for i in range(2):
            maze.add_wall( (1, i+7), "N")
        # wall F
        for i in range(3):
            maze.add_wall( (5, i+2), "N")
        # wall G
        for i in range(2):
            maze.add_wall( (i+6, 5), "W")
        # walls HIJK
        maze.add_wall( (6, 4), "N")
        maze.add_wall( (7, 4), "N")
        maze.add_wall( (8, 4), "W")
        maze.add_wall( (8, 3), "N")
        return maze

    def door_world(self):
        """ Grid world used in Experiment 2 """
        maze = self.create_maze_world(height= 6, width = 9)
        for i in range(maze.dims[0]):
            if i is not 3:
                maze.add_wall( (i, 6), "W")
        for j in range(6):
            if j is not 0:
                maze.add_wall( (2 , j), "N")
        maze.add_wall((2,2), "E")
        maze.add_wall((0,2), "E")
        return maze

    def door2_world(self):
        maze = self.create_maze_world(8, 8)
        for i in range(maze.width):
            if i is not 6: maze.add_wall([2, i], "N")
        for i in range(maze.width):
            if i is not 2: maze.add_wall([4, i], "N")
        return maze

    def tunnel_world(self):
        """ Grid world used in Experiment 3 """
        maze = self.create_maze_world(height= 5, width = 9)
        # vertical walls
        for i in range(maze.dims[0]):
            if i is not 2:
                maze.add_wall( (i, 6), "W")
                maze.add_wall( (i, 2), "W")
        # tunnel walls
        for j in range(2,6):
                maze.add_wall( (2 , j), "N")
                maze.add_wall( (2, j), "S")
        return maze

    def step_world(self):
        n = 5
        maze = self.create_maze_world(n,n)
        centre = np.array([(n-1)/2]*2, dtype = int)
        for i in (centre + [-1,+1]):
            for d in ['W','E']:
                maze.add_wall([i,centre[1]], d)
        for j in (centre + [-1,+1]):
            for d in ['N','S']:
                maze.add_wall([centre[0],j], d)
        return maze

    def left_right_door_world(self):
        maze = self.create_maze_world(height=6, width=7)
        for i in range(maze.width):
            if i not in [1, 5]: maze.add_wall([2, i], "N")
        return maze

    def simple(self):
        return self.create_maze_world(height=3, width=3)

