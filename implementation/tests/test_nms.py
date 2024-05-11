import unittest
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'implementation'))
from utils import non_max_suppression, intersection_over_union

class Non_Max_Suppression_Testing(unittest.TestCase):
    """
    Testing class for non-max suppression implementation 
    """
    def setUp(self):
        self.t1_bboxes = [
            [1, 0.5, 0.45, 0.4, 0.5, 0.01],
            [1, 0.5, 0.5, 0.2, 0.4, 0.49],
            [1, 0.25, 0.35, 0.3, 0.1, 0.7],
            [1, 0.1, 0.1, 0.1, 0.1, 0.05],
        ]
        self.t1_correct_nms = [[1, 0.25, 0.35, 0.3, 0.1, 0.7]]

        self.t2_bboxes = [
            [1, 0.5, 0.5, 0.6, 0.6, 1],
            [1, 0.5, 0.5, 0.575, 0.575, 0.8],
            [1, 0.25, 0.35, 0.3, 0.45, 0.7],
            [1, 0.1, 0.1, 0.1, 0.1, 0.05],
        ]
        self.t2_correct_nms = [
            [1, 0.5, 0.5, 0.6, 0.6, 1], 
            [1, 0.25, 0.35, 0.3, 0.45, 0.7]
        ]

        self.t3_bboxes = [
            [1, 0.5, 0.5, 0.6, 0.6, 1],
            [2, 0.5, 0.5, 0.75, 0.75, 0.9],
            [1, 0.25, 0.35, 0.3, 0.1, 0.8],
            [1, 0.1, 0.1, 0.1, 0.1, 0.05],
        ]
        self.t3_correct_nms = [
            [1, 0.5, 0.5, 0.6, 0.6, 1],
            [2, 0.5, 0.5, 0.75, 0.75, 0.9],
            [1, 0.25, 0.35, 0.3, 0.1, 0.8],
        ]

        self.t4_bboxes = [
            [1, 0.5, 0.5, 0.6, 0.6, 1],
            [1, 0.5, 0.5, 0.75, 0.75, 1],
            [1, 0.25, 0.35, 0.3, 0.45, 0.8],
            [1, 0.1, 0.1, 0.1, 0.1, 0.05],
        ]
        self.t4_correct_nms = [
            [1, 0.5, 0.5, 0.6, 0.6, 1],
            [1, 0.5, 0.5, 0.75, 0.75, 1],
            [1, 0.25, 0.35, 0.3, 0.45, 0.8],
        ]
    
    def test_filter_by_pred_threshold(self):
        nms = non_max_suppression(self.t1_bboxes)
        self.assertTrue(
            nms == self.t1_correct_nms,
            f'Test failed: {nms} != {self.t1_correct_nms}'
        )
    
    def test_filter_by_iou_threshold(self):
        nms = non_max_suppression(self.t2_bboxes)
        self.assertTrue(
            nms == self.t2_correct_nms, 
            f'Test failed: {nms} != {self.t2_correct_nms}'
        )
    
    def test_filter_including_diff_class(self):
        nms = non_max_suppression(self.t3_bboxes)
        self.assertTrue(
            nms == self.t3_correct_nms,
            f'Test failed: {nms} != {self.t3_correct_nms}'
        )
    
    def test_filter_by_iou_keep(self):
        nms = non_max_suppression(self.t4_bboxes)
        self.assertTrue(
            nms == self.t4_correct_nms,
            f'Test failed: {nms} != {self.t4_correct_nms}'
        )
    
if __name__ == '__main__':
    unittest.main()
