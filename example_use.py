from world.mazeworld import MazeWorld, WorldFactory
from world.multiworld import MultiWorldFactory
from world.socialworld import SocialWorldFactory
from world.pendulum import Pendulum
from strategy.empowerment import BlahutArimoto, VisitCount, VisitCountFast, BlahutArimotoTimeOptimal
from agent import EmpMaxAgent
from train import train_ma_agent, train_agent
from strategy.variational_empowerment import VariationalEmpowerment
import numpy as np 
import matplotlib.pyplot as plt
import time 


def example_1():
    """ builds maze world from original empowerment paper(https://uhra.herts.ac.uk/bitstream/handle/2299/1918/901933.pdf?sequence=1) and plots empowerment landscape.
        builds pendulum according to https://github.com/openai/gym/wiki/Pendulum-v0
    """
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

def example_2():
    """ Runs empowerment maximising agent running in a chosen grid world """
    np.random.seed(1)
    # maze
    n_step = 3
    f = WorldFactory()
    w = f.klyubin_world()#, tunnel_world()
    B = w.compute_transition()
    strategy = VisitCountFast()
    E = strategy.compute(world=w, T=B, n_step=n_step).reshape(-1)

    initpos = [1,3] # np.random.randint(w.dims[0], size=2)
    s = w._cell_to_index(initpos)

    # for reference
    emptymaze = MazeWorld(w.height, w.width)
    T = emptymaze.compute_transition()
    n_s, n_a, _ = T.shape

    # agent
    agent = EmpMaxAgent(alpha=0.1, gamma=0.9, T=T, n_step=n_step, n_samples=1000, det=1.)
    agent.s = s

    # training loop
    start = time.time()
    D_emp, D_mod, steps, tau, visited = train_agent(B, E, agent, w, n_s, n_a)
    print("elapsed seconds: %0.3f" % (time.time() - start))

    # some plotting
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(9, 6))
    #Amap = np.array([list(w.actions.values())[i] for i in agent.action_map])
    #ax[0, 0].quiver(np.arange(w.width) + .5, np.arange(w.height) + .5, Amap[:, 1].reshape(w.height, w.width), Amap[:, 0].reshape(w.height, w.width))

    w.plot(fig, ax[0, 0], colorMap= agent.E.reshape(*w.dims))
    ax[0, 0].set_title('subjective empowerment')
    print(f'min = {np.min(agent.E):.2f}, max = {np.max(agent.E):.2f}')

    w.plot(fig, ax[0,1], colorMap=visited.reshape(*w.dims))
    ax[0, 1].set_title('visited')

    Vmap = agent.value_map.reshape(*w.dims)
    w.plot(fig, ax[0, 2], colorMap= Vmap)
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

    w.plot(fig, ax[2, 0], colorMap= E.reshape(*w.dims))
    ax[2, 0].set_title('true empowerment')

    plt.show()

def example_3():
    """ compute empowerment landscape with neural networks"""
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(9, 9))
    f = WorldFactory()
    w = f.create_maze_world(6, 3)

    # Blahut as benchmark
    n_step = 1
    T = w.compute_transition()
    strategy = BlahutArimoto()

    start = time.time()
    E = strategy.compute(world=w, T=T, n_step=n_step)
    w.plot(fig, ax[0, 0], colorMap=E.reshape(w.dims))
    ax[0,0].set_title(f'{n_step}-step Blahut Arimoto {time.time() - start:.2f}s')

    # 3 step Blahut
    n_step = 3
    start = time.time()
    E = strategy.compute(world=w, T=T, n_step=n_step)
    w.plot(fig, ax[0, 1], colorMap=E.reshape(w.dims))
    ax[0, 1].set_title(f'{n_step}-step Blahut Arimoto {time.time() - start:.2f}s')

    # different scene
    w = f.simple()
    n_step = 1
    T = w.compute_transition()
    start = time.time()
    E = strategy.compute(world=w, T=T, n_step=n_step)
    w.plot(fig, ax[0, 2], colorMap=E.reshape(w.dims))
    ax[0, 2].set_title(f'{n_step}-step Blahut Arimoto {time.time() - start:.2f}s')

    # Variational empowerment
    w = f.create_maze_world(6, 3)
    n_step = 1
    T = w.compute_transition()
    strategy = VariationalEmpowerment(T.shape[0], T.shape[1], n_step=n_step)
    start = time.time()
    strategy.train_batch(world=w, T=T, n_step=n_step)
    E = strategy.compute(world=w, T=T, n_step=n_step)
    w.plot(fig, ax[1, 0], colorMap=E.reshape(w.dims))
    ax[1, 0].set_title(f'{n_step}-step Variational {time.time() - start:.2f}s')

    # 3 step variational
    n_step = 3
    start = time.time()
    strategy = VariationalEmpowerment(T.shape[0], T.shape[1], n_step=n_step)
    strategy.train_batch(world=w, T=T, n_step=n_step)
    E = strategy.compute(world=w, T=T, n_step=n_step)
    w.plot(fig, ax[1, 1], colorMap=E.reshape(w.dims))
    ax[1, 1].set_title(f'{n_step}-step Variational {time.time() - start:.2f}s')

    # different scene
    w = f.simple()
    n_step = 1
    T = w.compute_transition()
    start = time.time()
    strategy = VariationalEmpowerment(T.shape[0], T.shape[1], n_step=n_step)
    strategy.train_batch(world=w, T=T, n_step=n_step)
    E = strategy.compute(world=w, T=T, n_step=n_step)
    w.plot(fig, ax[1, 2], colorMap=E.reshape(w.dims))
    ax[1, 2].set_title(f'{n_step}-step Variational {time.time() - start:.2f}s')

    fig.tight_layout()


    plt.show()
    #plt.savefig("results/finalE.png")

def example_4():
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

    ax[0].pcolor(np.zeros(w.dims))
    w.plot_entities(fig, ax[0])
    ax[0].set_title(f'{n_step}-step E={E[idx[0]]:.2f} low')

    for j, agent in enumerate(w.agents):
        agent.set_s(w._index_to_location(idx[-1], j))

    ax[1].pcolor(np.zeros(w.dims))
    w.plot_entities(fig, ax[1])
    ax[1].set_title(f'{n_step}-step E={E[idx[-1]]:.2f} high')

    plt.show()

def example_5():
    """ compute empowerment landscape and value map for single agents in multi-agent scenario"""
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    np.random.seed(2)

    f = MultiWorldFactory()
    w = f.klyubin_2agents()

    visited = np.zeros(w.dims)

    steps = int(10000)
    for t in range(steps):
        s_all = w.interact()
        for s in s_all:
            cell = w._index_to_cell(s)
            visited[cell[0], cell[1]] += 1

        if t % 100 == 0:
            a0 = w.agents[0]
            if t%100==0:print(f'maxV = {np.max(a0.brain.value_map):.2f} minV= {np.min(a0.brain.value_map):.2f} at progress={t/steps*100:.1f}%')

        #if t % 100 == 0:
    a0 = w.agents[0]
    a1 = w.agents[1]

    w.plot(fig, ax[0, 0], colorMap=a0.brain.E.reshape(*w.dims))
    ax[0, 0].set_title(f'{a0.brain.n_step}-step empowerment agent 0')

    w.plot(fig, ax[0, 1], colorMap=visited.reshape(*w.dims))
    ax[0, 1].set_title(f'visited agents')

    traj = w.predict_trajectory(a0, a1, 4)
    w.plot(fig, ax[0, 2], colorMap=a0.brain.value_map.reshape(*w.dims), traj=traj)
    w.plot_entities(fig, ax[0, 2])
    ax[0, 2].set_title(f'value map agent 0 and prediction agent 1')

    plt.show()

def example_6():
    """ Runs empowerment maximising agent running in a chosen grid world """
    np.random.seed(2)
    # maze
    n_step = 1
    f = SocialWorldFactory()
    w = f.simple_2agents()
    B = w.compute_ma_transition(2)
    strategy = VisitCountFast()
    E = strategy.compute(world=w, T=B, n_step=n_step).reshape(-1)

    # for reference
    n_s, n_a, _ = B.shape

    # agent
    #w.agents[0].load_params()
    brain = w.agents[0].brain
    for agent in w.agents:
        agent.brain.decay = 5e-5

    # training loop
    start = time.time()
    D_emp, D_mod, steps, tau, visited, visited_config = train_ma_agent(B, E, brain, w, n_s, n_a)
    for agent in w.agents:
        agent.save_params()

    print("elapsed seconds: %0.3f" % (time.time() - start) )
    # some plotting
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(9, 6))

    w.plot(fig, ax[0, 0], colorMap= brain.E.reshape(w.dims[0], -1))
    ax[0, 0].set_title('subjective empowerment')
    print(f'min = {np.min(brain.E):.2f}, max = {np.max(brain.E):.2f}')

    w.plot(fig, ax[0,1], colorMap=visited.reshape(w.dims[0], -1))
    ax[0, 1].set_title('visited')

    w.plot(fig, ax[0,2], colorMap=brain.value_map.reshape(w.dims[0], -1))
    ax[0, 2].set_title('value map')

    ax[1, 0].scatter(brain.E, visited_config.reshape(n_s))
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

    w.plot(fig, ax[2, 0], colorMap= E.reshape(w.dims[0], -1))
    ax[2, 0].set_title('true empowerment')

    Vmap = brain.value_map
    idx = np.argsort(Vmap)
    for j, agent in enumerate(w.agents):
        agent.s = w._index_to_location(idx[0], j)

    w.plot(fig, ax[2, 1])
    w.plot_entities(fig, ax[2, 1])
    ax[2, 1].set_title(f'agent 1 low V')

    for j, agent in enumerate(w.agents):
        agent.s = w._index_to_location(idx[-1], j)

    w.plot(fig, ax[2, 2])
    w.plot_entities(fig, ax[2, 2])
    ax[2, 2].set_title(f'agent 1 high V')

    print(f'min = {np.min(Vmap):.2f}, max = {np.max(Vmap):.2f}')
    plt.show()


def example_7():
    f = SocialWorldFactory()
    w = f.simple_2agents()
    w.compute_ma_transition(2)
    for agent in w.agents:
        agent.load_params()

    s = [agent.s for agent in w.agents]
    c = w._location_to_index(s)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 4))
    for _ in range(100):
        c_ = w.interact(c, eval=True)
        traj = w.predict_trajectory(w.agents[0], 4)
        w.plot(fig, ax)
        w.plot_entities(fig, ax)
        for i in range(w.n_a):
            ax.plot(traj[:, i, 1]+.5, traj[:, i, 0]+.5)
            ax.text(traj[0, i, 1]+.5, traj[0, i, 0]+.5, f'{w.agents[i].brain.value_map[c_]:.0f}')

        plt.pause(1)
        c = c_




if __name__ == "__main__":
    from pathlib import Path
    Path("results").mkdir(parents=True, exist_ok=True)
    ## uncomment below to see examples
    # example_2()
    # example_3()
    # example_4()
    example_3()