import unittest
import torch
import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'implementation'))
from utils import onehot2index, getbo2

class Get_BO2_Testing(unittest.TestCase):
    """
    Testing class for ensuring the best of 2 is selected
    """
    def setUp(self):
        # Basic test to check for proper conversion 
        self.t1_preds = torch.Tensor([[[
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.55, 0.2, 0.3, 0.2, 0.9, 0.55, 0.2, 0.3, 0.2, 0.45],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.35, 0.6, 0.3, 0.2, 0.8, 0.35, 0.6, 0.3, 0.2, 0.96],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.8, 0.7, 0.2, 0.2, 0.7, 0.8, 0.7, 0.2, 0.2, 0.71]
        ]]])
        self.t1_correct_bo2s = torch.Tensor([[[
            [0.5500, 0.2000, 0.3000, 0.2000, 0.9000],
            [0.3500, 0.6000, 0.3000, 0.2000, 0.9600],
            [0.8000, 0.7000, 0.2000, 0.2000, 0.7100]
        ]]])
    
    def test_basic_convert(self):
        bo2_preds = getbo2(self.t1_preds)
        self.assertTrue(
            torch.equal(bo2_preds, self.t1_correct_bo2s), 
            f'Test failed: {bo2_preds} != {self.t1_correct_bo2s}'
        )

if __name__ == '__main__':
    unittest.main()
