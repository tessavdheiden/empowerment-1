import mazeworld
from mazeworld import MazeWorld, klyubin_world
from pendulum import Pendulum
from empowerment import BlahutArimoto, VisitCount, BlahutArimotoTimeOptimal, VisitCountFast
from agent import EmpMaxAgent
from variational_empowerment import VariationalEmpowerment
import numpy as np 
import matplotlib.pyplot as plt
import time 

def example_1():      
    """ builds maze world from original empowerment paper(https://uhra.herts.ac.uk/bitstream/handle/2299/1918/901933.pdf?sequence=1) and plots empowerment landscape. 
    """
    n_step = 5
    strategy = BlahutArimoto()
    maze = MazeWorld(10, 10)
    for i in range(6):
        maze.add_wall( (1, i), "N" )
    for i in range(2):
        maze.add_wall( (i+2, 5), "E")
        maze.add_wall( (i+2, 6), "E")
    maze.add_wall( (3, 6), "N")
    for i in range(2):
        maze.add_wall( (1, i+7), "N")
    for i in range(3):
        maze.add_wall( (5, i+2), "N")
    for i in range(2):
        maze.add_wall( (i+6, 5), "W")
    maze.add_wall( (6, 4), "N")
    maze.add_wall( (7, 4), "N")
    maze.add_wall( (8, 4), "W")
    maze.add_wall( (8, 3), "N")
    # compute the 5-step empowerment at each cell
    T = maze.compute_model()
    start = time.time()
    E = strategy.compute(world=maze, T=T, n_step=n_step)
    print(f"elapsed seconds: {time.time() - start:0.3f}")
    # plot the maze world
    maze.plot(colorMap=E)
    plt.title(f'{n_step}-step empowerment')
    plt.show()

def example_2():
    """ builds grid world with doors and plots empowerment landscape """
    maze = MazeWorld(8,8)
    strategy = VisitCountFast()
    for i in range(maze.width):
        if i is not 6 : maze.add_wall([2, i], "N")
    for i in range(maze.width):
        if i is not 2 : maze.add_wall([5, i], "N")
    n_step = 4
    T = maze.compute_model()
    start = time.time()
    E = strategy.compute(world=maze, T=T, n_step=n_step)
    print(f"elapsed seconds: {time.time() - start:0.3f}")
    # plot the maze world
    maze.plot(colorMap=E)
    plt.title(f'{n_step}-step empowerment')
    plt.show()

def example_3():
    """ Runs empowerment maximising agent running in a chosen grid world """
    # maze
    n_step = 3
    maze = mazeworld.door_world() # klyubin_world(), tunnel_world()
    B = maze.compute_model()
    strategy = VisitCountFast()
    E = strategy.compute(world=maze, T=B, n_step=n_step).reshape(-1)

    initpos = [4,4] # np.random.randint(maze.dims[0], size=2)
    s =  maze._cell_to_index(initpos)

    # for reference
    emptymaze = MazeWorld(maze.height, maze.width)
    T = emptymaze.compute_model()
    n_s, n_a, _ = T.shape

    # agent
    agent = EmpMaxAgent(alpha=0.1, gamma=0.9, T=T, n_step=n_step, n_samples=1000, det=1.)

    # training loop
    start = time.time()
    steps = int(10000)
    visited = np.zeros(maze.dims)
    tau = np.zeros(steps)
    D_emp = np.zeros(steps)
    D_mod = n_s*n_a*np.ones(steps)
    for t in range(steps):
        # append data for plotting
        tau[t] = agent.tau
        D_emp[t] = np.mean((E - agent.E)**2)
        D_mod[t] = D_mod[t] - np.sum(np.argmax(agent.T, axis=0) == np.argmax(B, axis=0))
        a = agent.act(s)
        pos = maze._index_to_cell(s)
        visited[pos[0],pos[1]] += 1
        s_ = maze.act(s,list(maze.actions.keys())[a])
        agent.update(s,a,s_)
        s = s_
    print("elapsed seconds: %0.3f" % (time.time() - start) )

    # some plotting
    plt.figure(1)
    plt.title("value and action map")
    Vmap = agent.value_map.reshape(*maze.dims)
    maze.plot(colorMap= Vmap )
    Amap = agent.action_map
    U = np.array([(1 if a == 2 else (-1 if a == 3 else 0)) for a in Amap]).reshape(maze.height, maze.width)
    V = np.array([(1 if a == 0 else (-1 if a == 1 else 0)) for a in Amap]).reshape(maze.height, maze.width)
    plt.quiver(np.arange(maze.width) + .5, np.arange(maze.height) + .5, U, V)
    plt.figure(2)
    plt.title("subjective empowerment")
    maze.plot(colorMap= agent.E.reshape(*maze.dims))
    plt.figure(3)
    plt.title("tau")
    plt.plot(tau)
    plt.figure(4)
    plt.scatter(agent.E, visited.reshape(n_s))
    plt.xlabel('true empowerment')
    plt.ylabel('visit frequency')
    plt.figure(5)
    plt.title("visited")
    maze.plot(colorMap=visited.reshape(*maze.dims))
    fig, ax1 = plt.subplots()
    red = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('MSE of empowerment map', color=red)
    ax1.plot(D_emp, color=red)
    ax1.tick_params(axis='y', labelcolor=red)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Model disagreement', color='tab:blue')
    ax2.plot(D_mod, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    plt.show()

def example_4():
    """ builds pendulum according to https://github.com/openai/gym/wiki/Pendulum-v0
    """
    pendulum = Pendulum(9,15)
    strategy = VisitCount()
    T = pendulum.compute_model()
    n_step = 5
    E = strategy.compute(world=pendulum, T=T, n_step=n_step)
    # plot the landscape
    pendulum.plot(colorMap=E)
    plt.title(f'{n_step}-step empowerment')
    plt.show()

def example_5():
    """ compute empowerment landscape with neural networks"""
    maze = MazeWorld(5, 5)
    n_step = 2
    start = time.time()
    T = maze.compute_model()
    strategy = VariationalEmpowerment(T.shape[0], T.shape[1], n_step=n_step)
    strategy.train_batch(world=maze, T=T, n_step=n_step)
    #strategy = VisitCount()
    E = strategy.compute(world=maze, T=T, n_step=n_step)
    print(f"elapsed seconds: {time.time() - start:0.3f}")
    maze.plot(colorMap=E)
    plt.title('%i-step empowerment' % n_step)
    plt.savefig("results/finalE.png")


if __name__ == "__main__":
    from pathlib import Path
    Path("results").mkdir(parents=True, exist_ok=True)
    ## uncomment below to see examples
    # example_2()
    # example_3()
    # example_4()
    example_5()