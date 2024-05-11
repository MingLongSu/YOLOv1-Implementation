import unittest 
import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'implementation'))
from utils import mean_average_precision

class Mean_Average_Precision_Testing(unittest.TestCase):
    """
    Testing class for mean average precision 
    """
    def setUp(self):
        # Setting acceptable error 
        self.epsilon = 1e-3

        # Testing for mAP == 1 True
        self.t1_preds = [
            [0, 0, 0.55, 0.2, 0.3, 0.2, 0.9],
            [0, 0, 0.35, 0.6, 0.3, 0.2, 0.8],
            [0, 0, 0.8, 0.7, 0.2, 0.2, 0.7],
        ]
        self.t1_gts = [
            [0, 0, 0.55, 0.2, 0.3, 0.2, 0.9],
            [0, 0, 0.35, 0.6, 0.3, 0.2, 0.8],
            [0, 0, 0.8, 0.7, 0.2, 0.2, 0.7],
        ]
        self.t1_correct_mAP = 1

        # Testing for mAP == 1 in a batch
        self.t2_preds = [
            [1, 0, 0.55, 0.2, 0.3, 0.2, 0.9],
            [0, 0, 0.35, 0.6, 0.3, 0.2, 0.8],
            [0, 0, 0.8, 0.7, 0.2, 0.2, 0.7],
        ]
        self.t2_gts = [
            [1, 0, 0.55, 0.2, 0.3, 0.2, 0.9],
            [0, 0, 0.35, 0.6, 0.3, 0.2, 0.8],
            [0, 0, 0.8, 0.7, 0.2, 0.2, 0.7],
        ]
        self.t2_correct_mAP = 1

        # Testing for all wrong class
        self.t3_preds = [
            [0, 1, 0.55, 0.2, 0.3, 0.2, 0.9],
            [0, 1, 0.35, 0.6, 0.3, 0.2, 0.8],
            [0, 1, 0.8, 0.7, 0.2, 0.2, 0.7],
        ]
        self.t3_gts = [
            [0, 0, 0.55, 0.2, 0.3, 0.2, 0.9],
            [0, 0, 0.35, 0.6, 0.3, 0.2, 0.8],
            [0, 0, 0.8, 0.7, 0.2, 0.2, 0.7],
        ]
        self.t3_correct_mAP = 0

        # Testing for one inaccurate box
        self.t4_preds = [
            [0, 0, 0.15, 0.25, 0.1, 0.1, 0.9],
            [0, 0, 0.35, 0.6, 0.3, 0.2, 0.8],
            [0, 0, 0.8, 0.7, 0.2, 0.2, 0.7],
        ]
        self.t4_gts = [
            [0, 0, 0.55, 0.2, 0.3, 0.2, 0.9],
            [0, 0, 0.35, 0.6, 0.3, 0.2, 0.8],
            [0, 0, 0.8, 0.7, 0.2, 0.2, 0.7],
        ]
        self.t4_correct_mAP = 5 / 18

    def test_all_correct_same_class(self):
        mAP = mean_average_precision(self.t1_preds, self.t1_gts)
        self.assertTrue(
            abs(mAP - self.t1_correct_mAP) < self.epsilon,
            f'Test failed: {mAP} != {self.t1_correct_mAP}'
        )

    def test_all_correct_various_class(self):
        mAP = mean_average_precision(self.t2_preds, self.t2_gts)
        self.assertTrue(
            abs(mAP - self.t2_correct_mAP) < self.epsilon,
            f'Test failed: {mAP} != {self.t2_correct_mAP}'
        )

    def test_all_incorrect_class(self):
        mAP = mean_average_precision(self.t3_preds, self.t3_gts)
        self.assertTrue(
            abs(mAP - self.t3_correct_mAP) < self.epsilon,
            f'Test failed: {mAP} != {self.t3_correct_mAP}'
        )

    def test_one_inaccurate_box(self):
        mAP = mean_average_precision(self.t4_preds, self.t4_gts)
        self.assertTrue(
            abs(mAP - self.t4_correct_mAP) < self.epsilon,
            f'Test failed: {mAP} != {self.t4_correct_mAP}'
        ) 

if __name__ == '__main__':
    unittest.main()
