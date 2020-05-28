import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from functools import reduce
import itertools

from strategy.empowerment_strategy import EmpowermentStrategy


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


class VariationalEmpowermentContinuous(EmpowermentStrategy):
    max_grad_norm = .5
    temp_min = 1
    temp = 15.

    def __init__(self, n_s, n_a, n_step):
        self.input_dim = n_s
        self.n_step = n_step
        self.source = Source(2, n_a**n_step).to(device)
        self.planner = Planner(2, n_a**n_step).to(device)
        self.params = list(self.source.parameters()) + list(self.planner.parameters())
        self.optimizer = optim.Adam(self.params, lr=3e-4)
        self.action_list = torch.from_numpy(np.arange(n_a**n_step)).float().view(1, -1).to(device)

    def compute(self, world, T, n_step, n_samples=int(1e3)):

        Bn = torch.from_numpy(world.Bn).float().to(device)
        n_s, n_aseq, _ = Bn.shape
        h, w = world.dims
        s = torch.arange(n_s).view(-1, 1).to(device)
        c = self._index_to_cell(s, h, w)
        #s_hot = self._index_to_one_hot(s, n_s).to(device).float()
        avg = torch.zeros(world.dims).to(device)
        #n_samples = max(n_samples, n_s*n_aseq)

        for _ in range(n_samples):
            with torch.no_grad():
                a, p_source = self.source.forward(c)
                s_ = self._propagate_state(Bn, s, a).max(1)[1].view(-1, 1)
                c_ = self._index_to_cell(s_, h, w)
                prob_planner = self.planner.forward(c, c_)
                avg += (torch.mul(prob_planner, a.detach()).sum(1) - torch.mul(p_source, a).sum(1)).view(world.dims)

        E = avg / n_samples #torch.log2(avg / n_samples)
        #self.q_x = prob.transpose(1, 0).cpu().numpy()
        return E.cpu().numpy()

    def _index_to_cell(self, s, height, width):
        cell = torch.stack([s / width, s % width], dim=1).float()
        return cell.squeeze(2)

    def _propagate_state(self, T, s, a):
        """ get updated state after action a
        s : batch of states, with each state being the index of grid position
        a : action sequence, implemented as one hot
        T : Transition matrix, starting in s, taking action sequence a, probability of reaching s' T(s', a, s)
        """
        n_s, n_a, _ = T.shape
        n_b, _ = s.shape
        T_rep = T.view(1, n_s * n_a, n_s).repeat(n_b, 1, 1)
        s_rep = s.view(-1, 1, 1).repeat(1, n_a * n_s, 1)
        batch = T_rep.gather(-1, s_rep).view(n_b, n_s, n_a).float()
        a = a.view(n_b, n_a, 1)

        out = torch.bmm(batch, a.float())  # out = [n_b, n_s, 1]
        return out.squeeze(2)

    def train_batch(self, world, T, n_step, n_samples=int(2.5e3)):
        """
        Compute the empowerment of a grid world with neural networks
        See eq 3 and 5 of https: // arxiv.org / pdf / 1710.05101.pdf
        """
        n_s, n_a, _ = T.shape
        h, w = world.dims
        Bn = torch.from_numpy(world.compute_nstep_transition_model(n_step=n_step)).float().to(device)

        s_all = torch.arange(n_s).view(-1, 1).to(device)

        anneal_rate = -np.log(self.temp) / n_samples
        start = time.time()
        for i in range(n_samples):
            temp = self.temp * np.exp( anneal_rate * i)
            s = s_all[torch.randperm(s_all.size()[0])].view(-1, 1)

            c = self._index_to_cell(s, h, w)
            a, p_source = self.source.forward(c, temp)

            s_ = self._propagate_state(Bn, s, a).max(1)[1].view(-1, 1)
            c_ = self._index_to_cell(s_, h, w)

            prob_planner = self.planner.forward(c, c_.detach())

            self.optimizer.zero_grad()
            error = -torch.mul(prob_planner, a.detach()).sum(1) + torch.mul(p_source, a).sum(1)
            loss = error.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
            self.optimizer.step()

            if i % 100 == 0:
                print(f"elapsed seconds: {time.time() - start:0.3f}, temp = {temp}, progress = {i/n_samples*100:.2f}%")
                start = time.time()
                # E = self.compute(world, T, n_step)
                # (fig, ax) = plt.subplots(1)
                # world.plot(fig, ax, colorMap=E)
                # plt.savefig(f"results/{i}.png")
                # plt.close(fig)

    def plot(self, fig, ax, states, world, n_step):

        a_list = [''.join(tup) for tup in list(itertools.product(world.actions.keys(), repeat=n_step))]
        h, w = world.dims
        states = torch.from_numpy(states).reshape(-1, 1)
        c = self._index_to_cell(states, h, w)
        with torch.no_grad():
            a, p_source = self.source.forward(c, temp=1)
        a = a.max(1)[1].cpu().numpy()
        for i, s in enumerate(states):
            ax.text(c[i, 1] + .5, c[i, 0] + .5, a_list[a[i]], horizontalalignment='center', verticalalignment='center')










