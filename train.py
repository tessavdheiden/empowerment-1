import numpy as np

def train_agent(B, E, agent, w, n_s, n_a, randomize=False):
    steps = int(10000)
    visited = np.zeros(w.dims)
    tau = np.zeros(steps)
    D_emp = np.zeros(steps)
    D_mod = n_s * n_a * np.ones(steps)
    for t in range(steps):
        if not randomize:
            a = agent.act(s)
            s_ = w.act(s, list(w.actions.keys())[a])
        else:
            s = np.random.randint(n_s)
            a = np.random.randint(n_a)
            s_ = np.argmax(w.T[:, a, s])
            agent.update_tau()

        if t%100==0:print(f'maxV = {np.max(agent.value_map):.2f} minV= {np.min(agent.value_map):.2f} at progress={t/steps*100:.1f}%')

        agent.update(s, a, s_)
        s = s_
        # append data for plotting
        tau[t] = agent.tau
        D_emp[t] = np.mean((E - agent.E) ** 2)
        D_mod[t] = D_mod[t] - np.sum(np.argmax(agent.T, axis=0) == np.argmax(B, axis=0))
        pos = w._index_to_cell(s_)
        visited[pos[0], pos[1]] += 1
    return D_emp, D_mod, steps, tau, visited

def train_ma_agent(B, E, brain, w, n_s, n_a, randomize=False):
    steps = int(100000)
    visited = np.zeros(w.dims)
    visited_config = np.zeros(n_s)
    tau = np.zeros(steps)
    D_emp = np.zeros(steps)
    D_mod = n_s * n_a * np.ones(steps)

    s = [agent.s for agent in w.agents]
    c = w._location_to_index(s)

    for t in range(steps):
        if not randomize:
            if t % 100 == 0: c = np.argmin(visited_config) # avoid overestimating unvisited
            c_ = w.interact(c)
        else:
            c = np.random.randint(n_s)
            a = brain.act(c)
            c_ = np.argmax(w.T[:, a, c])
            for agent in w.agents:
                agent.brain.update(c, a, c_)

        if t%100==0:print(f'maxV = {np.max(brain.value_map)} minV= {np.min(brain.value_map)} at progress={t/steps*100}%')

        c = c_
        # append data for plotting
        tau[t] = brain.tau
        D_emp[t] = np.mean((E - brain.E) ** 2)
        D_mod[t] = D_mod[t] - np.sum(np.argmax(brain.T, axis=0) == np.argmax(B, axis=0))
        for i, agent in enumerate(w.agents):
            s = w._index_to_location(c_, i)
            pos = w._index_to_cell(s)
            visited[pos[0], pos[1]] += 1
        visited_config[c_] += 1

    return D_emp, D_mod, steps, tau, visited, visited_config


