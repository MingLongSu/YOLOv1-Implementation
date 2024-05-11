import unittest
import torch
import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'implementation'))
from utils import onehot2index, getbo2

class Onehot_2_Index_Testing(unittest.TestCase):
    """
    Testing class for converting onehot encoded tensors to those with indices to 
    reference the C number of classes
    """
    def setUp(self):
        # Basic test to check for proper conversion 
        self.t1_onehots = torch.Tensor([[[
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.55, 0.2, 0.3, 0.2, 0.9],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.35, 0.6, 0.3, 0.2, 0.8],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.8, 0.7, 0.2, 0.2, 0.7]
        ]]])
        self.t1_correct_indexeds = torch.Tensor([[[
            [6, 0.55, 0.2, 0.3, 0.2, 0.9],
            [16, 0.35, 0.6, 0.3, 0.2, 0.8],
            [11, 0.8, 0.7, 0.2, 0.2, 0.7]
        ]]])
    
    def test_basic_convert(self):
        indexeds = onehot2index(self.t1_onehots)
        getbo2(
            torch.Tensor([[[
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.55, 0.2, 0.3, 0.2, 0.9, 0.55, 0.2, 0.3, 0.2, 0.45],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.35, 0.6, 0.3, 0.2, 0.8, 0.35, 0.6, 0.3, 0.2, 0.96],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.8, 0.7, 0.2, 0.2, 0.7, 0.8, 0.7, 0.2, 0.2, 0.71]
            ]]])
        )
        self.assertTrue(
            torch.equal(indexeds, self.t1_correct_indexeds), 
            f'Test failed: {indexeds} != {self.t1_correct_indexeds}'
        )

if __name__ == '__main__':
    unittest.main()
