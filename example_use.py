from world.mazeworld import MazeWorld, WorldFactory
from world.multiworld import MultiWorldFactory
from world.socialworld import SocialWorldFactory
from world.pendulum import Pendulum
from strategy.empowerment import BlahutArimoto, VisitCount, VisitCountFast, BlahutArimotoTimeOptimal
from agent import EmpMaxAgent
from strategy.variational_empowerment import VariationalEmpowerment
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
    T = maze.compute_transition()
    start = time.time()
    E = strategy.compute(world=maze, T=T, n_step=n_step)
    print(f"elapsed seconds: {time.time() - start:0.3f}")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
    # plot the maze world
    maze.plot(fig, ax, colorMap=E.reshape(maze.dims))
    plt.title(f'{n_step}-step empowerment')
    plt.show()

def example_2():
    """ builds grid world with doors and plots empowerment landscape """
    f = WorldFactory()
    strategy = VisitCount()
    maze = f.door2_world()
    n_step = 4
    T = maze.compute_transition()
    start = time.time()
    E = strategy.compute(world=maze, T=T, n_step=n_step)
    print(f"elapsed seconds: {time.time() - start:0.3f}")
    # plot the maze world
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
    maze.plot(fig, ax, colorMap=E.reshape(maze.dims))
    plt.title(f'{n_step}-step empowerment')
    plt.show()

def example_3():
    """ Runs empowerment maximising agent running in a chosen grid world """
    np.random.seed(1)
    # maze
    n_step = 3
    f = WorldFactory()
    maze = f.klyubin_world()#, tunnel_world()
    B = maze.compute_transition()
    strategy = VisitCountFast()
    E = strategy.compute(world=maze, T=B, n_step=n_step).reshape(-1)

    initpos = [1,3] # np.random.randint(maze.dims[0], size=2)
    s = maze._cell_to_index(initpos)

    # for reference
    emptymaze = MazeWorld(maze.height, maze.width)
    T = emptymaze.compute_transition()
    n_s, n_a, _ = T.shape

    # agent
    agent = EmpMaxAgent(alpha=0.1, gamma=0.9, T=T, n_step=n_step, n_samples=1000, det=1.)
    agent.s = s

    # training loop
    start = time.time()
    D_emp, D_mod, steps, tau, traj, visited = train_agent(B, E, agent, maze, n_s, n_a)
    print("elapsed seconds: %0.3f" % (time.time() - start) )
    # some plotting
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(9, 6))
    #Amap = np.array([list(maze.actions.values())[i] for i in agent.action_map])
    #ax[0, 0].quiver(np.arange(maze.width) + .5, np.arange(maze.height) + .5, Amap[:, 1].reshape(maze.height, maze.width), Amap[:, 0].reshape(maze.height, maze.width))

    maze.plot(fig, ax[0, 0], colorMap= agent.E.reshape(*maze.dims))
    ax[0, 0].set_title('subjective empowerment')
    print(f'min = {np.min(agent.E):.2f}, max = {np.max(agent.E):.2f}')

    maze.plot(fig, ax[0,1], colorMap=visited.reshape(*maze.dims))
    ax[0, 1].set_title('visited')

    Vmap = agent.value_map.reshape(*maze.dims)
    maze.plot(fig, ax[0, 2], colorMap= Vmap)
    ax[0, 2].set_title('value map')
    print(f'min = {np.min(Vmap):.2f}, max = {np.max(Vmap):.2f}')

    ax[1, 1].set_title("tau")
    ax[1, 1].plot(tau)

    ax[1, 0].scatter(agent.E, visited.reshape(n_s))
    ax[1, 0].set_xlabel('true empowerment')
    ax[1, 0].set_ylabel('visit frequency')

    red = 'tab:red'
    ax[1, 2].plot(D_emp, color=red)
    ax[1, 2].set_xlabel('time')
    ax[1, 2].set_ylabel('MSE of empowerment map', color=red)
    ax[1, 2].tick_params(axis='y', labelcolor=red)

    ax[1, 2] = ax[1, 2].twinx()
    ax[1, 2].set_ylabel('Model disagreement', color='tab:blue')
    ax[1, 2].plot(D_mod, color='tab:blue')
    ax[1, 2].tick_params(axis='y', labelcolor='tab:blue')

    maze.plot(fig, ax[2, 0], colorMap= E.reshape(*maze.dims))
    ax[2, 0].set_title('true empowerment')

    plt.show()

def example_4():
    """ builds pendulum according to https://github.com/openai/gym/wiki/Pendulum-v0
    """
    pendulum = Pendulum(9,15)
    strategy = VisitCountFast()
    start = time.time()
    T = pendulum.compute_transition()
    n_step = 5
    E = strategy.compute(world=pendulum, T=T, n_step=n_step)
    print(f"elapsed seconds: {time.time() - start:0.3f}")
    # plot the landscape
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
    pendulum.plot(fig, ax, colorMap=E.reshape(pendulum.dims))
    plt.title(f'{n_step}-step empowerment')
    plt.show()

def example_5():
    """ compute empowerment landscape with neural networks"""
    maze = MazeWorld(5, 5)
    n_step = 3
    start = time.time()
    T = maze.compute_transition()
    strategy = VariationalEmpowerment(T.shape[0], T.shape[1], n_step=n_step)
    strategy = BlahutArimoto()
    #strategy.train_batch(world=maze, T=T, n_step=n_step)
    E = strategy.compute(world=maze, T=T, n_step=n_step)
    print(f"elapsed seconds: {time.time() - start:0.3f}")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    maze.plot(fig, ax, colorMap=E.reshape(maze.dims))
    ax.set_title('%i-step empowerment' % n_step)
    plt.show()
    #plt.savefig("results/finalE.png")

def example_6():
    """ compute empowerment landscape for scenerios"""
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 8))
    n_step = 3
    strategy = VisitCount()
    f = WorldFactory()

    w = f.klyubin_world()
    T = w.compute_transition()
    E = strategy.compute(world=w, T=T, n_step=n_step)
    w.plot(fig, ax[0, 0], colorMap=E.reshape(w.dims))
    ax[0, 0].set_title(f'{n_step}-step klyubin')

    w = f.door_world()
    T = w.compute_transition()
    E = strategy.compute(world=w, T=T, n_step=n_step)
    w.plot(fig, ax[1, 0], colorMap=E.reshape(w.dims))
    ax[1, 0].set_title(f'{n_step}-step door')

    w = f.door2_world()
    T = w.compute_transition()
    E = strategy.compute(world=w, T=T, n_step=n_step)
    w.plot(fig, ax[2, 0], colorMap=E.reshape(w.dims))
    ax[2, 0].set_title(f'{n_step}-step door2')

    w = f.step_world()
    T = w.compute_transition()
    E = strategy.compute(world=w, T=T, n_step=n_step)
    w.plot(fig, ax[0, 1], colorMap=E.reshape(w.dims))
    ax[0, 1].set_title(f'{n_step}-step step')

    w = f.tunnel_world()
    T = w.compute_transition()
    E = strategy.compute(world=w, T=T, n_step=n_step)
    w.plot(fig, ax[1, 1], colorMap=E.reshape(w.dims))
    ax[1, 1].set_title(f'{n_step}-step tunnel')

    w = Pendulum(9, 15)
    T = w.compute_transition()
    E = strategy.compute(world=w, T=T, n_step=n_step)
    w.plot(fig, ax[2, 1], colorMap=E.reshape(w.dims))
    ax[2, 1].set_title(f'{n_step}-step pendulum')

    plt.show()

def example_7():
    """ compute combined empowerment landscapes for all agents in multi-agent scenario"""
    np.random.seed(3)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))
    n_step = 1
    strategy = BlahutArimotoTimeOptimal()

    f = MultiWorldFactory()

    w = f.simple_2agents()
    start = time.time()

    T = w.compute_ma_transition(w.n_a, det=1.)
    E = strategy.compute(world=w, T=T, n_step=n_step)
    print(f"elapsed seconds: {time.time() - start:0.3f}")

    # Only for plotting, positions need to be resetted
    idx = np.argsort(E)
    for j, agent in enumerate(w.agents):
        agent.set_s(w._index_to_location(idx[0], j))

    w.plot(fig, ax[0], colorMap=np.zeros(w.dims), show_entities=True)
    ax[0].set_title(f'{n_step}-step E={E[idx[0]]:.2f} low')

    for j, agent in enumerate(w.agents):
        agent.set_s(w._index_to_location(idx[-1], j))

    w.plot(fig, ax[1], colorMap=np.zeros(w.dims), show_entities=True)
    ax[1].set_title(f'{n_step}-step E={E[idx[-1]]:.2f} high')

    plt.show()

def example_8():
    """ compute empowerment landscape and value map for single agents in multi-agent scenario"""
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    np.random.seed(2)

    f = MultiWorldFactory()
    w = f.simple_2agents()

    visited0 = np.zeros(w.dims)
    visited1 = np.zeros(w.dims)

    steps = int(1000)
    for t in range(steps):
        w.interact()
        cell = w.agents[0].get_cell()
        visited0[cell[0], cell[1]] += 1

        cell = w.agents[1].get_cell()
        visited1[cell[0], cell[1]] += 1

        if t % 500 == 0:
            a0 = w.agents[0]
            idx = np.argsort(a0.brain.value_map)
            print(f'low V ={a0.brain.value_map[idx[0]]:.2f} high V ={a0.brain.value_map[idx[-1]]:.2f}')

        #if t % 100 == 0:
    a0 = w.agents[0]
    a1 = w.agents[1]

    w.plot(fig, ax[0, 0], colorMap=a0.brain.E.reshape(*w.dims))
    ax[0, 0].set_title(f'{a0.brain.n_step}-step empowerment agent 0')

    w.plot(fig, ax[0, 1], colorMap=visited0.reshape(*w.dims))
    ax[0, 1].set_title(f'visited agent 0')

    traj = w.predict_trajectory(a0, a1, 4)
    w.plot(fig, ax[0, 2], colorMap=a0.brain.value_map.reshape(*w.dims), traj=traj, show_entities=True)
    ax[0, 2].set_title(f'value map agent 0 and prediction agent 1')

    w.plot(fig, ax[1, 0], colorMap=a1.brain.E.reshape(*w.dims))
    ax[1, 0].set_title(f'{a1.brain.n_step}-step empowerment agent 1')

    w.plot(fig, ax[1, 1], colorMap=visited1.reshape(*w.dims))
    ax[1, 1].set_title(f'visited agent 1')

    traj = w.predict_trajectory(a1, a0, 4)
    w.plot(fig, ax[1, 2], colorMap=a1.brain.value_map.reshape(*w.dims), traj=traj, show_entities=True)
    ax[1, 2].set_title(f'value map agent 1 and prediction agent 0')

            #plt.pause(.001)

    plt.show()

def example_9():
    """ compute empowerment landscape and value map for collaborating agents in multi-agent scenario"""
    np.random.seed(1)

    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(16, 8))
    f = SocialWorldFactory()
    w = f.simple_2agents()
    w.compute_ma_transition(2)

    visited0 = np.zeros(w.dims)
    visited1 = np.zeros(w.dims)

    steps = int(10000)
    for t in range(steps):
        start = time.time()
        w.interact_ma(2)

        cell = w.agents[0].get_cell()
        visited0[cell[0], cell[1]] += 1

        cell = w.agents[1].get_cell()
        visited1[cell[0], cell[1]] += 1

        if t % 500 == 0:
            a0 = w.agents[0]
            idx = np.argsort(a0.brain.value_map)
            print(f'low V ={a0.brain.value_map[idx[0]]:.2f} high V ={a0.brain.value_map[idx[-1]]:.2f}')
            print(f"progress: {t}/{steps}={t / steps * 100:.1f}% elapsed seconds: {time.time() - start:0.3f}")

            a0 = w.agents[0]
            a1 = w.agents[1]

            w.plot(fig, ax[0, 2], colorMap=visited0.reshape(*w.dims))
            ax[0, 2].set_title(f'visited agent 0')

            w.plot(fig, ax[1, 2], colorMap=visited1.reshape(*w.dims))
            ax[1, 2].set_title(f'visited agent 1')

    actions=[]
    states=[]
    for j, agent in enumerate(w.agents):
        states.append(agent.s)
        actions.append(agent.action)

    idx = np.argsort(a0.brain.E)
    for j, agent in enumerate(w.agents):
        agent.s = w._index_to_location(idx[0], j)

    w.plot(fig, ax[0, 0])
    ax[0, 0].set_title(f'agent 0 low E')

    for j, agent in enumerate(w.agents):
        agent.s = w._index_to_location(idx[-1], j)

    w.plot(fig, ax[0, 1])
    ax[0, 1].set_title(f'agent 0 high E')

    idx = np.argsort(a0.brain.value_map)
    for j, agent in enumerate(w.agents):
        agent.s = w._index_to_location(idx[0], j)

    w.plot(fig, ax[0, 3])
    ax[0, 3].set_title(f'agent 0 low V')

    for j, agent in enumerate(w.agents):
        agent.s = w._index_to_location(idx[-1], j)

    w.plot(fig, ax[0, 4])
    ax[0, 4].set_title(f'agent 0 high V')

    # second agent
    idx = np.argsort(a1.brain.E)
    for j, agent in enumerate(w.agents):
        agent.s = w._index_to_location(idx[0], j)

    w.plot(fig, ax[1, 0])
    ax[1, 0].set_title(f'agent 1 low E')

    for j, agent in enumerate(w.agents):
        agent.s = w._index_to_location(idx[-1], j)

    w.plot(fig, ax[1, 1])
    ax[1, 1].set_title(f'agent 1 high E')

    idx = np.argsort(a1.brain.value_map)
    for j, agent in enumerate(w.agents):
        agent.s = w._index_to_location(idx[0], j)

    w.plot(fig, ax[1, 3])
    ax[1, 3].set_title(f'agent 1 low V')

    for j, agent in enumerate(w.agents):
        agent.s = w._index_to_location(idx[-1], j)

    w.plot(fig, ax[1, 4])
    ax[1, 4].set_title(f'agent 1 high V')

    # some statistics
    ax[2, 0].pcolor(a0.brain.value_map.reshape(w.dims[0], -1))
    ax[2, 0].set_title(f"agent 0 value map")

    ax[2, 1].pcolor(a1.brain.value_map.reshape(w.dims[0], -1))
    ax[2, 1].set_title(f"agent 1 value map")

    for j, agent in enumerate(w.agents):
        agent.s = states[j]
        agent.action = actions[j]

    plt.pause(.001)

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
        a = agent.act(s)
        s_ = maze.act(s, list(maze.actions.keys())[a])

        # s = np.random.randint(n_s)
        # a = np.random.randint(n_a)
        # s_ = np.argmax(maze.T[:, a, s])

        if t%100==0:print(f'maxV = {np.max(agent.value_map)} minV= {np.min(agent.value_map)} at progress={t/steps*100}%')

        agent.update(s, a, s_)
        agent.s = s_
        # append data for plotting
        tau[t] = agent.tau
        D_emp[t] = np.mean((E - agent.E) ** 2)
        D_mod[t] = D_mod[t] - np.sum(np.argmax(agent.T, axis=0) == np.argmax(B, axis=0))
        pos = maze._index_to_cell(s_)
        visited[pos[0], pos[1]] += 1
        traj.append(pos)
    return D_emp, D_mod, steps, tau, traj, visited

def train_ma_agent(B, E, brain, maze, n_s, n_a):
    steps = int(10000)
    visited = np.zeros(maze.dims)
    visited_config = np.zeros(n_s)
    tau = np.zeros(steps)
    D_emp = np.zeros(steps)
    D_mod = n_s * n_a * np.ones(steps)
    traj = []

    for t in range(steps):
        c = np.random.randint(n_s)
        a = brain.act(c)
        c_ = np.argmax(maze.T[:, a, c])

        if t%100==0:print(f'maxV = {np.max(brain.value_map)} minV= {np.min(brain.value_map)} at progress={t/steps*100}%')
        brain.update(c, a, c_)

        # append data for plotting
        tau[t] = brain.tau
        D_emp[t] = np.mean((E - brain.E) ** 2)
        D_mod[t] = D_mod[t] - np.sum(np.argmax(brain.T, axis=0) == np.argmax(B, axis=0))
        s = maze._index_to_location(c_, 0)
        pos = maze._index_to_cell(s)
        visited[pos[0], pos[1]] += 1
        visited_config[c_] += 1
        traj.append(pos)
    return D_emp, D_mod, steps, tau, traj, visited, visited_config

def example_10():
    """ Runs empowerment maximising agent running in a chosen grid world """
    np.random.seed(1)
    # maze
    n_step = 1
    f = SocialWorldFactory()
    maze = f.simple_2agents()
    B = maze.compute_ma_transition(2)
    strategy = VisitCountFast()
    E = strategy.compute(world=maze, T=B, n_step=n_step).reshape(-1)

    initpos = [1,3] # np.random.randint(maze.dims[0], size=2)
    s = maze._cell_to_index(initpos)

    # for reference
    n_s, n_a, _ = B.shape

    # agent
    agent = maze.agents[0].brain

    # training loop
    start = time.time()
    D_emp, D_mod, steps, tau, traj, visited, visited_config = train_ma_agent(B, E, agent, maze, n_s, n_a)

    print("elapsed seconds: %0.3f" % (time.time() - start) )
    # some plotting
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(9, 6))
    #Amap = np.array([list(maze.actions.values())[i] for i in agent.action_map])
    #ax[0, 0].quiver(np.arange(maze.width) + .5, np.arange(maze.height) + .5, Amap[:, 1].reshape(maze.height, maze.width), Amap[:, 0].reshape(maze.height, maze.width))

    maze.plot(fig, ax[0, 0], colorMap= agent.E.reshape(maze.dims[0], -1))
    ax[0, 0].set_title('subjective empowerment')
    print(f'min = {np.min(agent.E):.2f}, max = {np.max(agent.E):.2f}')

    maze.plot(fig, ax[0,1], colorMap=visited.reshape(maze.dims[0], -1))
    ax[0, 1].set_title('visited')

    ax[1, 0].scatter(agent.E, visited_config.reshape(n_s))
    ax[1, 0].set_xlabel('true empowerment')
    ax[1, 0].set_ylabel('visit frequency')

    ax[1, 1].set_title("tau")
    ax[1, 1].plot(tau)

    red = 'tab:red'
    ax[1, 2].plot(D_emp, color=red)
    ax[1, 2].set_xlabel('time')
    ax[1, 2].set_ylabel('MSE of empowerment map', color=red)
    ax[1, 2].tick_params(axis='y', labelcolor=red)

    ax[1, 2] = ax[1, 2].twinx()
    ax[1, 2].set_ylabel('Model disagreement', color='tab:blue')
    ax[1, 2].plot(D_mod, color='tab:blue')
    ax[1, 2].tick_params(axis='y', labelcolor='tab:blue')

    maze.plot(fig, ax[2, 0], colorMap= E.reshape(maze.dims[0], -1))
    ax[2, 0].set_title('true empowerment')

    Vmap = agent.value_map
    idx = np.argsort(Vmap)
    for j, agent in enumerate(maze.agents):
        agent.s = maze._index_to_location(idx[0], j)

    maze.plot(fig, ax[2, 1], show_entities=True)
    ax[2, 1].set_title(f'agent 1 low V')

    for j, agent in enumerate(maze.agents):
        agent.s = maze._index_to_location(idx[-1], j)

    maze.plot(fig, ax[2, 2], show_entities=True)
    ax[2, 2].set_title(f'agent 1 high V')

    print(f'min = {np.min(Vmap):.2f}, max = {np.max(Vmap):.2f}')
    plt.show()


if __name__ == "__main__":
    from pathlib import Path
    Path("results").mkdir(parents=True, exist_ok=True)
    ## uncomment below to see examples
    # example_2()
    # example_3()
    # example_4()
    example_9()