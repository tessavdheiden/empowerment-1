import unittest
import torch

from strategy.variational_empowerment import *

class VETest(unittest.TestCase):
    def test_index_to_cell(self):
        ve = VariationalEmpowerment(n_s=6, n_a=3, n_step=1)
        batch_size = 10
        h, w = 3, 4
        n_s = int(w*h)
        s = torch.randint(0, n_s, (batch_size,)).view(-1, 1)
        c = ve._index_to_cell(s, h, w)
        self.assertTrue(torch.all(c[:, 0] < h))
        self.assertTrue(torch.all(c[:, 1] < w))
        self.assertEqual(c.shape[0], batch_size)
        self.assertEqual(c.shape[1], 2)


if __name__ == "__main__":
    unittest.main()

