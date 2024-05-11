import torch

from collections import Counter

class Compose(object):
    """
    Class to define applied transformations to image data
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, lbl):
        for t in self.transform:
            img, lbl = t(img), lbl
        return img, lbl

class Singleton(type):
    """
    Class to define singleton-style objects
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class PascalVOCClasses(metaclass=Singleton):
    """
    Class used to represent the mapping from integer label to string label

    Attributes
    -------
    lab2classes : dict
        Mapping of label representations from integer to string

    Methods
    -------
    __call__()
        Returns a mapping of integer label to string label representation
    """

    def __init__(self):
        self.lab2class = {
            0: 'airplane',
            1: 'bicycle',
            2: 'bird',
            3: 'boat',
            4: 'bottle',
            5: 'bus',
            6: 'car',
            7: 'cat',
            8: 'chair',
            9: 'cow',
            10: 'dining table',
            11: 'dog',
            12: 'horse',
            13: 'motor bike',
            14: 'person',
            15: 'potted plant',
            16: 'sheep',
            17: 'sofa',
            18: 'train', 
            19: 'tv monitor'
        }

    def __call__(self):
        return self.lab2class
    
def intersection_over_union(bboxes_1: torch.Tensor, bboxes_2: torch.Tensor, 
                            box_format: str='midpoint', epsilon=1e-6):
    """
    Calculates the intersection over union of two given bounding boxes assuming unnorm

    Parameters
    -------
    bboxes_1 : torch.Tensor 
        Bounding box details of first box
    bboxes_2 : torch.Tensor 
        Bounding box details of second box
    box_format : str
        Details that format of the input tensor
    epsilon : float
        Amount of error added for numerical stability
    
    Returns
    -------
    iou : torch.Tensor 
        Computes the area of intersection divided by area of union of given bounding boxes
    """
    # Must compute the coordinates if given in normalized midpoint form
    if (box_format == 'midpoint'):
        # Define the coordinate details for bounding box 1
        bboxes_1_x = bboxes_1[..., 0:1]
        bboxes_1_y = bboxes_1[..., 1:2]
        bboxes_1_w = bboxes_1[..., 2:3]
        bboxes_1_h = bboxes_1[..., 3:4]

        # Compute the top left (tl) and bottom right (br) coordinates for bounding box 1
        bboxes_1_tl_x = bboxes_1_x - (bboxes_1_w / 2)
        bboxes_1_tl_y = bboxes_1_y - (bboxes_1_h / 2)
        bboxes_1_br_x = bboxes_1_x + (bboxes_1_w / 2)
        bboxes_1_br_y = bboxes_1_y + (bboxes_1_h / 2)

        # Define the coordinate details for bounding box 2
        bboxes_2_x = bboxes_2[..., 0:1]
        bboxes_2_y = bboxes_2[..., 1:2]
        bboxes_2_w = bboxes_2[..., 2:3]
        bboxes_2_h = bboxes_2[..., 3:4]

        # Compute the top left (tl) and bottom right (br) coordinates for bounding box 2
        bboxes_2_tl_x = bboxes_2_x - (bboxes_2_w / 2)
        bboxes_2_tl_y = bboxes_2_y - (bboxes_2_h / 2)
        bboxes_2_br_x = bboxes_2_x + (bboxes_2_w / 2)
        bboxes_2_br_y = bboxes_2_y + (bboxes_2_h / 2)
    else:
        # Collect the corner coordinates of first bounding box
        bboxes_1_tl_x = bboxes_1[..., 0:1]
        bboxes_1_tl_y = bboxes_1[..., 1:2]
        bboxes_1_br_x = bboxes_1[..., 2:3]
        bboxes_1_br_y = bboxes_1[..., 3:4]

        # Collect the corner coordinates of the second bounding box
        bboxes_2_tl_x = bboxes_2[..., 0:1]
        bboxes_2_tl_y = bboxes_2[..., 1:2]
        bboxes_2_br_x = bboxes_2[..., 2:3]
        bboxes_2_br_y = bboxes_2[..., 3:4]

    # Get coordinates of intersection
    intersection_tl_x = torch.max(bboxes_1_tl_x, bboxes_2_tl_x)
    intersection_tl_y = torch.max(bboxes_1_tl_y, bboxes_2_tl_y)
    intersection_br_x = torch.min(bboxes_1_br_x, bboxes_2_br_x)
    intersection_br_y = torch.min(bboxes_1_br_y, bboxes_2_br_y)

    # Compute area of intersection
    intersection_diff_x = (intersection_br_x - intersection_tl_x).clamp(0)
    intersection_diff_y = (intersection_br_y - intersection_tl_y).clamp(0)
    area_of_intersection = intersection_diff_x * intersection_diff_y

    # Compute area of union
    bboxes_1_diff_x = bboxes_1_br_x - bboxes_1_tl_x
    bboxes_1_diff_y = bboxes_1_br_y - bboxes_1_tl_y
    area_of_bboxes_1 = torch.abs(bboxes_1_diff_x * bboxes_1_diff_y)
    bboxes_2_diff_x = bboxes_2_br_x - bboxes_2_tl_x
    bboxes_2_diff_y = bboxes_2_br_y - bboxes_2_tl_y
    area_of_bboxes_2 = torch.abs(bboxes_2_diff_x * bboxes_2_diff_y)
    area_of_union = area_of_bboxes_1 + area_of_bboxes_2 - area_of_intersection + epsilon

    # Compute intersection over union
    ious = area_of_intersection / area_of_union

    return ious

def non_max_suppression(bboxes: list,  threshold_iou: float=0.5, threshold_pred: float=0.5):
    """
    Performs non-maximal suppression algorithm to retrieve best prediction

    Parameters
    -------
    bboxes : list
        Bounding box predictions found along grid with each item in list of shape (5,)
    threshold_iou : float
        Minimum coverage to be considered a good prediction based on IOU
    threshold_pred : float
        Minimum confidence to be considered a good prediction independent of IOU
    
    Returns
    -------
    nms_bboxes : list
        Bounding boxes after non-maximal suppression
    """

    # Filter based on confidence scoring of bounding box by threshold of prediction
    confident_bboxes = [
        bbox
        for bbox in bboxes
        if bbox[-1] > threshold_pred
    ]
    confident_bboxes_sorted = sorted(
        confident_bboxes, key=lambda bbox: bbox[-1], 
        reverse=True
    )

    # Filter based on iou using threshold iou
    bboxes_post_nms = []
    while confident_bboxes_sorted:
        top_bbox = confident_bboxes_sorted.pop(0)
        bboxes_post_nms.append(top_bbox)

        confident_bboxes_sorted = [
            bbox 
            for bbox in confident_bboxes_sorted 
            if (bbox[0] != top_bbox[0] or 
                (intersection_over_union(
                    torch.tensor(top_bbox[1:5]),
                    torch.tensor(bbox[1:5]),
                    box_format='corner'
                ) < threshold_iou))
        ]
        
    return bboxes_post_nms

def mean_average_precision(bboxes_pred: list, bboxes_gt: list,
                           threshold_iou: float=0.5, n_classes: int=20, epsilon=1e-6):
    """
    Computes the mean average precision (mAP) across the n_classes

    Parameters
    -------
    bboxes_pred : list
        Bounding box information of the predictions bounding boxes
    bboxes_gt : list
        Bounding box information of the ground truth bounding boxes
    threshold_iou : float
        Minimum confidence to be considered a good prediction based on IOU
    n_classes : int
        Number of classes the YOLO is trained on
    
    Returns
    -------
    mAP : float
        Mean average precision across the n_classes
    """
    # List to collect average precisions of each class
    class_APs = []

    # Iterate over each class to individually compute AP
    for classification in range(n_classes):
        detects_of_class = []
        gts_of_class = []

        # Collect predictions and ground truths of particular class
        for pred in bboxes_pred:
            if (pred[1] == classification):
                detects_of_class.append(pred)
        for gt in bboxes_gt:
            if (gt[1] == classification):
                gts_of_class.append(gt)

        # Get number of bboxes for this class
        n_bboxes_of_class = len(gts_of_class)
        bboxes_of_class = Counter([gts[0] for gts in gts_of_class])
        for k, v in bboxes_of_class.items():
            bboxes_of_class[k] = torch.zeros(v)

        # If there are no bounding boxes for this class, then skip
        if (n_bboxes_of_class == 0):
            continue

        # Sort both bounding boxes based on prediction bounding box confidences
        sort_permutation = sorted(
            range(len(detects_of_class)), key=lambda k: detects_of_class[k][-1],
            reverse=True
        )
        detects_of_class_sorted = [detects_of_class[index] for index in sort_permutation]
        gts_of_class_sorted = [gts_of_class[index] for index in sort_permutation]

        # Define true positives (TP), false positives (FP)
        TP = torch.zeros(len(detects_of_class_sorted))
        FP = torch.zeros(len(detects_of_class_sorted))

        # Iterate over detections 
        for detect_index, detect in enumerate(detects_of_class_sorted):
            # Get the ground truths with the same training index as predictions
            gts_wsame_train_index = [
                bbox for bbox in gts_of_class_sorted if bbox[0] == detect[0]
            ]

            # Get number of ground truths with corresponding training index
            n_gts = len(gts_wsame_train_index)
            best_iou = 0
            best_gt_index = None

            # Compute the intersection over unions to see for match
            for gts_index, gt in enumerate(gts_wsame_train_index):
                iou = intersection_over_union(
                    torch.tensor(detect[2:-1]),
                    torch.tensor(gt[2:-1]),
                    box_format='midpoint', 
                    epsilon=epsilon
                )

                # Update the best found iou and corresponding ground truth
                if (iou > best_iou):
                    best_iou = iou
                    best_gt_index = gts_index
            
            # If considered valid detection
            if (best_iou > threshold_iou):
                # Check if ground truth has already been used
                if (bboxes_of_class[detect[0]][best_gt_index] == 0):
                    TP[detect_index] = 1
                    bboxes_of_class[detect[0]][best_gt_index] = 1
                else:
                    FP[detect_index] = 1
            else:
                FP[detect_index] = 1

        # Calculate recall and precision
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recall = TP_cumsum / (n_bboxes_of_class + epsilon)
        recalls = torch.cat((torch.tensor([0]), recall))
        precision = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precision))
        class_APs.append(torch.trapz(precisions, recalls))
    return sum(class_APs) / len(class_APs)

def getbo2(bboxes_preds: torch.Tensor):
    """
    Gets the best out of the two predicted bounding boxes in a given cell 

    bboxes_preds : torch.Tensor
        Bounding boxes of shape (-1, S, S, C + B*5) that will be converted to (-1, S, S, C + 5)

    Returns
    -------
    bboxes_preds_best : torch.Tensor
        Returns the bounding boxes of shape (-1, S, S, C + 5) where B becomes 1 due to having
        taken the most confident bounding box
    """
    confidence_scores = torch.concat(
        [bboxes_preds[..., 24:25].unsqueeze(0), bboxes_preds[..., 29:30].unsqueeze(0)],
        dim=0
    )
    _, best_confidences = torch.max(confidence_scores, dim=0)
    bboxes_preds_best = (
        (1 - best_confidences) * bboxes_preds[..., 20:25] + 
        best_confidences * bboxes_preds[..., 25:30]
    )
    return bboxes_preds_best

def onehot2index(bboxes: torch.Tensor, n_classes: int=20):
    """
    Converts the YoloV1 model or ground truths with shape (-1, S, S, C + 5) to (-1, S, S, 1 + 5)\
    
    Parameters
    -------
    bboxes : torch.Tensor
        Bounding boxes of shape (-1, S, S, C + B*5) that will be converted to (-1, S, S, 1 + 5)
    n_classes : int
        Details the original number of classes to represent the one-hot encoding 

    Returns
    -------
    bboxes_indexed : torch.Tensor
        Returns the bounding boxes of shape (-1, S, S, 1 + 5) where C is collapsed to 1 by 
        converting from one-hot encoding to index indicator
    """
    batch_size, S1, S2, _ = bboxes.shape
    bboxes_indexed = torch.zeros((batch_size, S1, S2, 1 + 5))
    bboxes_indexed[..., :1] = torch.argmax(bboxes[..., :n_classes], dim=-1).unsqueeze(-1)
    bboxes_indexed[..., 1:] = bboxes[..., n_classes:]
    return bboxes_indexed

def tensors2lists(bboxes_pred: torch.Tensor, bboxes_gt: torch.Tensor):
    """
    Converts the YoloV1 model output and ground truth tensors to lists of 
    tensors with corresponding training indices

    Parameters
    -------
    bboxes_pred : torch.Tensor 
        Bounding box predictions stored as a tensor with shape (-1, S, S, 1 + 5)
    bboxes_gt: torch.Tensor
        Bounding box ground truths stored as a tensor also with shape (-1, S, S, 1 + 5)
    
    Returns
    -------
    bboxes_list_pred : list
        Bounding box information of the predictions for a given image stored as a list
    bboxes_list_gt : list
        Bounding box information of the ground truths for a given image as a list
    """
    bboxes_list_pred = []
    bboxes_list_gt = []

    # Retrieving the grid size of the datas
    batch_size, S1, S2, _ = bboxes_pred.shape 

    # Iterate over grid cells to add all predictions to a list
    for batch_index in range(batch_size):
        for i in range(S1):
            for j in range(S2):
                bboxes_list_pred += torch.concat(
                    [torch.Tensor([batch_index]), bboxes_pred[batch_index, i, j, :6]], 
                    dim=0
                ) 

    # Iterate over grid cells to add all ground truths to a list
    for batch_index in range(batch_size):
        for i in range(S1):
            for j in range(S2):
                bboxes_list_gt += torch.concat(
                    [torch.Tensor([batch_index]), bboxes_gt[batch_index, i, j, :6]],
                    dim=0
                )

    return bboxes_list_pred, bboxes_list_gt
