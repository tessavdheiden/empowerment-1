import unittest
import torch

from world.multiworld import MultiWorld, MultiWorldFactory

class MWTest(unittest.TestCase):
    def test_one_hot(self):
        f = MultiWorldFactory()
        w = f.simple_2agents()




if __name__ == "__main__":
    unittest.main()
