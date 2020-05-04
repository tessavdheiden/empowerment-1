import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

from empowerment_strategy import EmpowermentStrategy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Source(nn.Module):
    def __init__(self, input_dim, categorical_dim):
        super(Source, self).__init__()
        self.input_dim = input_dim
        self.categorical_dim = categorical_dim
        self.fc = nn.Linear(self.input_dim, 200)
        self.a_head = nn.Linear(200, self.categorical_dim)

    def forward(self, x, temp=1):
        x = F.relu(self.fc(x))
        action_score = self.a_head(x)
        #z = torch.nn.functional.one_hot(prob.max(1)[1], num_classes=self.categorical_dim).view(-1, self.categorical_dim)
        return F.gumbel_softmax(action_score, hard=True, dim=-1, tau=temp), F.gumbel_softmax(action_score, hard=False, dim=-1, tau=temp)

class Planner(nn.Module):
    def __init__(self, input_dim, categorical_dim):
        super(Planner, self).__init__()
        self.categorical_dim = categorical_dim
        self.input_dim = input_dim
        self.fc = nn.Linear(self.input_dim, 200)
        self.fc_ = nn.Linear(self.input_dim, 200)
        self.hidden = nn.Linear(400, 400)
        self.a_head = nn.Linear(400, self.categorical_dim)
        self.temp = 1

    def forward(self, x, x_):
        x = F.relu(self.fc(x))
        x_ = F.relu(self.fc_(x_))
        cat = torch.cat([x, x_], dim=-1)
        h = F.relu(self.hidden(cat))
        action_score = self.a_head(h)
        return F.softmax(action_score, dim=-1)


class VariationalEmpowerment(EmpowermentStrategy):
    max_grad_norm = .5
    temp_min = .1
    anneal_rate = 3e-7#0.00003
    temp = 10.
    def __init__(self, n_s, n_a, n_step):
        self.input_dim = n_s
        self.source = Source(n_s, n_a**n_step).to(device)
        self.planner = Planner(n_s, n_a**n_step).to(device)
        self.optimizer_planner = optim.Adam(self.planner.parameters(), lr=3e-4)
        self.optimizer_source = optim.Adam(self.source.parameters(), lr=3e-4)
        self.params = list(self.source.parameters()) + list(self.planner.parameters())
        self.optimizer = optim.Adam(self.params, lr=3e-4)
        self.action_list = torch.from_numpy(np.arange(n_a**n_step)).float().view(1, -1).to(device)

    def compute(self, world, T, n_step, n_samples=int(1e3)):
        n_states, _, _ = T.shape
        Bn = world.compute_nstep_transition_model(n_step)
        _, n_actionseq, _ = Bn.shape

        T = torch.from_numpy(Bn).float().to(device)

        S = torch.arange(n_states).view(-1, 1).to(device)
        S_hot = one_hot_batch_matrix(S, n_states).to(device).float()
        avg = torch.zeros(world.dims).to(device)
        n_samples = max(n_samples, n_states*n_actionseq)

        for j in range(n_samples):
            with torch.no_grad():
                z, prob = self.source.forward(S_hot)
                s_ = get_s_next_from_s_batch_matrix(T, S, z)
                prob_ = self.planner.forward(S_hot, s_)
                avg += (torch.mul(prob_, z.detach()).sum(1)).view(world.dims) #- torch.mul(prob, z).sum(1)).view(world.dims)
                if j % 499 == 0:
                    for i, s in enumerate(S):
                        if prob_[i].max(0)[1] != z[i].max(0)[1] and (s == 8 or s == 18):
                        #if s == 8 or s == 18:
                            print(f"s={s.item()}, s_={s_[i].max(0)[1]}, p={prob_[i].max(0)[1]}, z_source={z[i].max(0)[1]}, \n"
                                  f"probs = {prob[i].data.cpu().numpy()}, \n"
                                  f"probs_planning = {prob_[i].data.cpu().numpy()}")

        E = torch.log2(avg / n_samples)
        self.q_x = prob.transpose(1, 0).cpu().numpy()
        return E.cpu().numpy()

    def train_batch(self, world, T, n_step, n_samples=int(5e5)):
        """
         Compute the empowerment of a grid world with neural networks
         See eq 3 and 5 of https: // arxiv.org / pdf / 1710.05101.pdf
         """

        n_states, n_actions, _ = T.shape
        Bn = world.compute_nstep_transition_model(n_step=n_step)

        T = torch.from_numpy(Bn).float().to(device)
        states = torch.arange(n_states).view(-1, 1).to(device)

        start = time.time()
        for i in range(n_samples):
            #if i % 100 == 0 and i != 0: self.temp = np.maximum(self.temp * np.exp(-self.anneal_rate * i), self.temp_min)
            S = states[torch.randperm(states.size()[0])]
            S_hot = one_hot_batch_matrix(S, self.input_dim).float()
            Z, prob = self.source.forward(S_hot.float())

            S_ = get_s_next_from_s_batch_matrix(T, S, Z)

            prob_ = self.planner.forward(S_hot, S_.detach())

            self.optimizer.zero_grad()
            error = -torch.mul(prob_, Z.detach()).sum(1) + torch.mul(prob, Z).sum(1)
            loss = error.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
            self.optimizer.step()

            if i % 10000 == 0:
                print(f"elapsed seconds: {time.time() - start:0.3f}")
                start = time.time()
                E = self.compute(world, T, n_step)
                (fig, ax) = plt.subplots(1)
                world.plot(colorMap=E, figax=(fig, ax))
                self.plot(world.width, world.height)
                plt.savefig(f"results/{i}.png")
                plt.close(fig)


def one_hot_vector(x, input_dim):
    return torch.zeros(1, input_dim).scatter_(1, x.view(-1, 1), 1)


def one_hot_batch_matrix(x, input_dim):
    return torch.nn.functional.one_hot(x, num_classes=input_dim).view(x.shape[0], input_dim)


def get_s_next_from_one_hot(T, s_hot, z):
    assert s_hot.shape[1] > 1
    t = (T * s_hot.float()).sum(2)
    return torch.mm(t, z.float().T).T


def get_s_next(T, s, z):
    assert s.shape[1] == 1
    idx = s.unsqueeze(0).repeat(T.shape[0], T.shape[1], 1)
    t = T[:, :, :].gather(2, idx)
    return torch.mm(t.squeeze(2), z.T).T


def get_s_next_from_one_hot_batch_matrix(T, s_hot, z):
    assert s_hot.shape[0] > 1 and z.shape[0] > 1
    n_s, n_a, _ = T.shape
    n_b, _ = s_hot.shape
    t = T.view(-1, n_s * n_a, n_s)
    t = t.repeat(n_b, 1, 1)
    s_hot = s_hot.view(n_b, n_s, 1)

    batch = torch.bmm(t.float(), s_hot.float()) # out = [n_b, n_s * n_a, 1]
    batch = batch.squeeze(2).view(n_b, n_s, n_a)
    z = z.view(n_b, n_a, 1)

    out = torch.bmm(batch, z.float()) # out = [n_b, n_s, 1]
    return out.squeeze(2)


def get_s_next_from_s_batch_matrix(T, s, z):
    n_s, n_a, _ = T.shape
    n_b, _ = s.shape
    t = T.view(1, n_s * n_a, n_s).repeat(n_b, 1, 1)
    batch = t.gather(-1, s.view(-1, 1, 1).repeat(1, n_a * n_s, 1)).view(n_b, n_s, n_a).float()
    z = z.view(n_b, n_a, 1)

    out = torch.bmm(batch, z.float()) # out = [n_b, n_s, 1]
    return out.squeeze(2)



