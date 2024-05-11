import torch
import unittest
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'implementation'))
from utils import intersection_over_union

class Intersection_Over_Union_Testing(unittest.TestCase):
    """
    Testing class for intersection over union function implementation
    """
    def setUp(self):
        # Setting acceptable error (epsilon)
        self.epsilon = 1e-3

        # Testing basic
        self.t1_bbox1 = torch.tensor([0.8, 0.1, 0.2, 0.2])
        self.t1_bbox2 = torch.tensor([0.9, 0.2, 0.2, 0.2])
        self.t1_correct_iou = 1/7

        # Testing slight overlap
        self.t2_bbox1 = torch.tensor([0.95, 0.6, 0.5, 0.2])
        self.t2_bbox2 = torch.tensor([0.95, 0.7, 0.3, 0.2])
        self.t2_correct_iou = 3 / 13

        # Testing no overlapping bounding boxes
        self.t3_bbox1 = torch.tensor([0.25, 0.15, 0.3, 0.1])
        self.t3_bbox2 = torch.tensor([0.25, 0.35, 0.3, 0.1])
        self.t3_correct_iou = 0

        # Testing 100% overlapping bounding boxes
        self.t4_bbox1 = torch.tensor([0.5, 0.5, 0.2, 0.2])
        self.t4_bbox2 = torch.tensor([0.5, 0.5, 0.2, 0.2])
        self.t4_correct_iou = 1

    def test_overlapping(self):
        iou = intersection_over_union(self.t1_bbox1, self.t1_bbox2)
        self.assertTrue(
            torch.abs(iou - self.t1_correct_iou) < self.epsilon,
            f'Test failed: {iou} != {self.t1_correct_iou}'
        )

    def test_slight_overlappping(self):
        iou = intersection_over_union(self.t2_bbox1, self.t2_bbox2)
        self.assertTrue(
            torch.abs(iou - self.t2_correct_iou) < self.epsilon, 
            f'Test failed: {iou} != {self.t2_correct_iou}'    
        )

    def test_no_overlapping(self):
        iou = intersection_over_union(self.t3_bbox1, self.t3_bbox2)
        self.assertTrue(
            torch.abs(iou - self.t3_correct_iou) < self.epsilon,
            f'Test failed: {iou} != {self.t3_correct_iou}'
        )

    def test_full_overlapping(self):
        iou = intersection_over_union(self.t4_bbox1, self.t4_bbox2)
        self.assertTrue(
            torch.abs(iou - self.t4_correct_iou) < self.epsilon,
            f'Test failed: {iou} != {self.t4_correct_iou}'
        )
    
if __name__ == '__main__':
    unittest.main()
