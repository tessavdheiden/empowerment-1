from mazeworld import MazeWorld, WorldFactory
from multiworld import MultiWorldFactory
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
    f = WorldFactory()
    maze = f.klyubin_world()
    # compute the 5-step empowerment at each cell
    T = maze.compute_model()
    start = time.time()
    E = strategy.compute(world=maze, T=T, n_step=n_step)
    print(f"elapsed seconds: {time.time() - start:0.3f}")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
    # plot the maze world
    maze.plot(fig, ax, colorMap=E)
    plt.title(f'{n_step}-step empowerment')
    plt.show()

def example_2():
    """ builds grid world with doors and plots empowerment landscape """
    f = WorldFactory()
    maze = f.door2_world()
    n_step = 4
    T = maze.compute_model()
    start = time.time()
    E = strategy.compute(world=maze, T=T, n_step=n_step)
    print(f"elapsed seconds: {time.time() - start:0.3f}")
    # plot the maze world
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
    maze.plot(fig, ax, colorMap=E)
    plt.title(f'{n_step}-step empowerment')
    plt.show()

def example_3():
    """ Runs empowerment maximising agent running in a chosen grid world """
    # maze
    n_step = 3
    f = WorldFactory()
    maze = f.klyubin_world()#, tunnel_world()
    B = maze.compute_model()
    strategy = VisitCountFast()
    E = strategy.compute(world=maze, T=B, n_step=n_step).reshape(-1)

    initpos = [1,3] # np.random.randint(maze.dims[0], size=2)
    s =  maze._cell_to_index(initpos)

    # for reference
    emptymaze = MazeWorld(maze.height, maze.width)
    T = emptymaze.compute_model()
    n_s, n_a, _ = T.shape

    # agent
    agent = EmpMaxAgent(alpha=0.1, gamma=0.9, T=T, n_step=n_step, n_samples=1000, det=1.)
    agent.s = s

    # training loop
    start = time.time()
    D_emp, D_mod, steps, tau, traj, visited = train_agent(B, E, agent, maze, n_s, n_a)
    print("elapsed seconds: %0.3f" % (time.time() - start) )
    # some plotting
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(9, 3))
    Vmap = agent.value_map.reshape(*maze.dims)
    maze.plot(fig, ax[0, 0], colorMap= Vmap)
    ax[0, 0].set_title('value map')
    Amap = np.array([list(maze.actions.values())[i] for i in agent.action_map])
    ax[0, 0].quiver(np.arange(maze.width) + .5, np.arange(maze.height) + .5, Amap[:, 1].reshape(maze.height, maze.width), Amap[:, 0].reshape(maze.height, maze.width))

    maze.plot(fig, ax[0, 1], colorMap= agent.E.reshape(*maze.dims))
    ax[0, 1].set_title('subjective empowerment')

    ax[0, 2].set_title("tau")
    ax[0, 2].plot(tau)

    ax[1, 0].scatter(agent.E, visited.reshape(n_s))
    ax[1, 0].set_xlabel('true empowerment')
    ax[1, 0].set_ylabel('visit frequency')

    maze.plot(fig, ax[1,1], colorMap=visited.reshape(*maze.dims))
    ax[1, 1].set_title('visited')

    red = 'tab:red'
    ax[1, 2].plot(D_emp, color=red)
    ax[1, 2].set_xlabel('time')
    ax[1, 2].set_ylabel('MSE of empowerment map', color=red)
    ax[1, 2].tick_params(axis='y', labelcolor=red)

    ax[1, 2] = ax[1, 2].twinx()
    ax[1, 2].set_ylabel('Model disagreement', color='tab:blue')
    ax[1, 2].plot(D_mod, color='tab:blue')
    ax[1, 2].tick_params(axis='y', labelcolor='tab:blue')
    plt.show()


def train_agent(B, E, agent, maze, n_s, n_a):
    steps = int(10000)
    visited = np.zeros(maze.dims)
    tau = np.zeros(steps)
    D_emp = np.zeros(steps)
    D_mod = n_s * n_a * np.ones(steps)
    traj = []
    for t in range(steps):
        s = agent.s
        # append data for plotting
        tau[t] = agent.tau
        D_emp[t] = np.mean((E - agent.E) ** 2)
        D_mod[t] = D_mod[t] - np.sum(np.argmax(agent.T, axis=0) == np.argmax(B, axis=0))
        a = agent.act(s)
        pos = maze._index_to_cell(s)
        traj.append(pos)
        visited[pos[0], pos[1]] += 1
        s_ = maze.act(s, list(maze.actions.keys())[a])
        agent.update(s, a, s_)
        agent.s = s_
    return D_emp, D_mod, steps, tau, traj, visited


def example_4():
    """ builds pendulum according to https://github.com/openai/gym/wiki/Pendulum-v0
    """
    pendulum = Pendulum(9,15)
    strategy = VisitCountFast()
    start = time.time()
    T = pendulum.compute_model()
    n_step = 5
    E = strategy.compute(world=pendulum, T=T, n_step=n_step)
    print(f"elapsed seconds: {time.time() - start:0.3f}")
    # plot the landscape
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
    pendulum.plot(fig, ax, colorMap=E)
    plt.title(f'{n_step}-step empowerment')
    plt.show()

def example_5():
    """ compute empowerment landscape with neural networks"""
    maze = MazeWorld(5, 5)
    n_step = 3
    start = time.time()
    T = maze.compute_model()
    strategy = VariationalEmpowerment(T.shape[0], T.shape[1], n_step=n_step)
    strategy = BlahutArimoto()
    #strategy.train_batch(world=maze, T=T, n_step=n_step)
    E = strategy.compute(world=maze, T=T, n_step=n_step)
    print(f"elapsed seconds: {time.time() - start:0.3f}")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    maze.plot(fig, ax, colorMap=E)
    ax.set_title('%i-step empowerment' % n_step)
    plt.show()
    #plt.savefig("results/finalE.png")

def example_6():
    """ compute empowerment landscape for multiple agents"""
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 8))
    n_step = 1
    strategy = VisitCount()
    f = WorldFactory()

    klyubin = f.klyubin_world()
    T = klyubin.compute_model()
    E = strategy.compute(world=klyubin, T=T, n_step=n_step)
    klyubin.plot(fig, ax[0, 0], colorMap=E)
    ax[0, 0].set_title(f'{n_step}-step klyubin')

    door = f.door_world()
    T = door.compute_model()
    E = strategy.compute(world=door, T=T, n_step=n_step)
    door.plot(fig, ax[1, 0], colorMap=E)
    ax[1, 0].set_title(f'{n_step}-step door')

    door = f.door2_world()
    T = door.compute_model()
    E = strategy.compute(world=door, T=T, n_step=n_step)
    door.plot(fig, ax[2, 0], colorMap=E)
    ax[2, 0].set_title(f'{n_step}-step door2')

    step = f.step_world()
    T = step.compute_model()
    E = strategy.compute(world=step, T=T, n_step=n_step)
    step.plot(fig, ax[0, 1], colorMap=E)
    ax[0, 1].set_title(f'{n_step}-step step')

    tunnel = f.tunnel_world()
    T = tunnel.compute_model()
    E = strategy.compute(world=tunnel, T=T, n_step=n_step)
    tunnel.plot(fig, ax[1, 1], colorMap=E)
    ax[1, 1].set_title(f'{n_step}-step tunnel')

    pendulum = Pendulum(9, 15)
    T = pendulum.compute_model()
    E = strategy.compute(world=pendulum, T=T, n_step=n_step)
    pendulum.plot(fig, ax[2, 1], colorMap=E)
    ax[2, 1].set_title(f'{n_step}-step pendulum')

    plt.show()

def example_7():
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
    np.random.seed(3)

    f = MultiWorldFactory()
    multiagent = f.klyubin_world_ma()
    steps = int(10000)
    for t in range(steps):
        multiagent.interact()
    multiagent.plot(fig, ax[0, 0], colorMap=multiagent.agents[0].value_map.reshape(*multiagent.dims))
    multiagent.plot(fig, ax[0, 1], colorMap=multiagent.agents[0].E.reshape(*multiagent.dims))
    multiagent.plot(fig, ax[0, 2], colorMap=multiagent.agents[0].visited.reshape(*multiagent.dims))

    multiagent.plot(fig, ax[1, 0], colorMap=multiagent.agents[1].value_map.reshape(*multiagent.dims))
    multiagent.plot(fig, ax[1, 1], colorMap=multiagent.agents[1].E.reshape(*multiagent.dims))
    multiagent.plot(fig, ax[1, 2], colorMap=multiagent.agents[1].visited.reshape(*multiagent.dims))

    ax[0, 0].set_title(f'value map agent 0')
    ax[0, 1].set_title(f'{multiagent.agents[0].n_step}-step empowerment agent 0')
    ax[0, 2].set_title(f'visited agent 0')

    ax[1, 0].set_title(f'value map agent 1')
    ax[1, 1].set_title(f'{multiagent.agents[1].n_step}-step empowerment agent 1')
    ax[1, 2].set_title(f'visited agent 1')
    #plt.pause(.0001)

    plt.show()


if __name__ == "__main__":
    from pathlib import Path
    Path("results").mkdir(parents=True, exist_ok=True)
    ## uncomment below to see examples
    # example_2()
    # example_3()
    # example_4()
    example_7()