import unittest
import torch

from strategy.variational_empowerment import *
from strategy.variational_empowerment_continuous import *

class VETest(unittest.TestCase):
    def test_index_to_one_hot(self):
        ve = VariationalEmpowerment(n_s=3, n_a=4, n_step=1)
        t = torch.tensor([[1], [3], [1]])
        y = ve._index_to_one_hot(t, 4)
        r = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]])
        self.assertTrue(torch.all(y.eq(r)))

    def test_get_next_batch(self):
        ve = VariationalEmpowerment(n_s=6, n_a=3, n_step=1)
        batch_size = 32
        T = torch.tensor([[[0, 0, 0], [1, 1, 0]], [[1, 0, 0], [0, 0, 1]], [[0, 1, 1], [0, 0, 0]]])
        s = torch.tensor([1]).long().view(-1, 1).repeat(batch_size, 1)
        s_hot = ve._index_to_one_hot(s, 3)
        z = torch.tensor([[1, 0]]).repeat(batch_size, 1)
        s_ = get_s_next_from_one_hot_batch_matrix(T, s_hot, z)
        r = torch.tensor([0, 0, 1])
        self.assertTrue(torch.all(s_[0].eq(r)))
        self.assertTrue(torch.all(s_[-1].eq(r)))

    def test_propagate(self):
        batch_size = 32
        ve = VariationalEmpowerment(n_s=6, n_a=3, n_step=1)
        T = torch.tensor([[[0, 0, 0], [1, 1, 0]], [[1, 0, 0], [0, 0, 1]], [[0, 1, 1], [0, 0, 0]]])
        s = torch.tensor([1]).long().view(-1, 1).repeat(batch_size, 1)
        #s_hot = one_hot_batch_matrix(s, 3)
        z = torch.tensor([[1, 0]]).repeat(batch_size, 1)
        s_ = ve._propagate_state(T, s, z)
        r = torch.tensor([0, 0, 1])
        self.assertTrue(torch.all(s_[0].eq(r)))
        self.assertTrue(torch.all(s_[-1].eq(r)))

    def test_index_to_cell(self):
        ve = VariationalEmpowermentContinuous(n_s=6, n_a=3, n_step=1)
        batch_size = 10
        h, w = 3, 4
        n_s = int(w*h)
        s = torch.randint(0, n_s, (batch_size,)).view(-1, 1)
        c = ve._index_to_cell(s, h, w)
        self.assertTrue(torch.all(c[:, 0] < h))
        self.assertTrue(torch.all(c[:, 1] < w))
        self.assertEqual(c.shape[0], batch_size)
        self.assertEqual(c.shape[1], 2)


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

if __name__ == "__main__":
    unittest.main()
