import unittest
import numpy as np
import torch

from variational_empowerment import *

class VETest(unittest.TestCase):
    def test_numpy_to_torch(self):
        n = np.array([1])
        t = numpy_to_torch(n)
        self.assertIsInstance(t, torch.Tensor)

        n = np.random.rand(10, 10)
        t = numpy_to_torch(n)
        self.assertEqual(t.shape, (10, 10))

    def test_one_hot(self):
        t = torch.tensor([1])
        y = one_hot_vector(t, 4)
        self.assertEqual(y.shape, (1, 4))
        r = torch.tensor([[0, 1, 0, 0]])
        self.assertTrue(torch.all(y.eq(r)))

    def test_one_hot_batch(self):
        t = torch.tensor([[1], [3], [1], [2]])
        y = one_hot_batch_matrix(t, 4)
        r = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
        self.assertTrue(torch.all(y.eq(r)))

    def test_get_s_next(self):
        T = torch.tensor([[[0, 0, 0], [1, 1, 0]] , [[1, 0, 0], [0, 0, 1]], [[0, 1, 1], [0, 0, 0]]])
        s = torch.tensor([1]).long().view(-1, 1)
        s_hot = one_hot_vector(s, 3)
        z = torch.tensor([[1, 0]]) # right
        s_ = get_s_next_from_one_hot(T, s_hot, z)
        r = torch.tensor([0, 0, 1])
        self.assertTrue(torch.all(s_.eq(r)))
        s_ = get_s_next(T, s, z)
        self.assertTrue(torch.all(s_.eq(r)))

    def test_get_s_next_s_hot_batch(self):
        batch_size = 32
        T = torch.tensor([[[0, 0, 0], [1, 1, 0]], [[1, 0, 0], [0, 0, 1]], [[0, 1, 1], [0, 0, 0]]])
        s = torch.tensor([1]).long().view(-1, 1).repeat(batch_size, 1)
        s_hot = one_hot_batch_matrix(s, 3)
        z = torch.tensor([[1, 0]]).repeat(batch_size, 1)
        s_ = get_s_next_from_one_hot_batch_matrix(T, s_hot, z)
        r = torch.tensor([0, 0, 1])
        self.assertTrue(torch.all(s_[0].eq(r)))
        self.assertTrue(torch.all(s_[-1].eq(r)))

if __name__ == "__main__":
    unittest.main()
