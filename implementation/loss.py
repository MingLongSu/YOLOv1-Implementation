import torch
import torch.nn as nn

from utils import intersection_over_union

class YOLOv1Loss(nn.Module):
    def __init__(self, grid_size: int=7, n_bounding_boxes: int=2, n_classes: int=20,
                 lambda_coord: float=5, lambda_noobj: float=0.5):
        super(YOLOv1Loss, self).__init__()
        self.grid_size = grid_size
        self.n_bounding_boxes = n_bounding_boxes
        self.n_classes = n_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction='sum')
        self.epsilon = 1e-6

    def forward(self, preds: torch.Tensor, gts: torch.Tensor):
        # Since output of model flattened, then reshape to paper specifications
        preds = torch.reshape(
            preds, 
            (-1, self.grid_size, self.grid_size, 
             self.n_bounding_boxes * 5 + self.n_classes)
        )

        # Compute intersection over union between predictions and ground_truths
        iou1 = intersection_over_union(preds[..., 20:24], gts[..., 20:24]).unsqueeze(0)
        iou2 = intersection_over_union(preds[..., 25:29],  gts[..., 20:24]).unsqueeze(0)

        # Concat the two to later determine the best bounding box from the two ious
        ious = torch.cat([iou1, iou2], dim=0)
        _, best_ious = torch.max(ious, dim=0)
        I_obj_ij = gts[..., 24:25]

        # Collect best bounding boxes into a single tensor
        best_bbox_preds = I_obj_ij * (
            ((1 - best_ious) * preds[..., 20:24] + best_ious * preds[..., 25:29])
        )

        # Use identity obj (I_obj_ij) to filter out ground truths
        gts_bboxes = I_obj_ij * gts[..., 20:24]

        # Compute bounding box (x, y, w, h) loss
        best_bbox_preds[..., 2:4] = torch.sign(best_bbox_preds[..., 2:4]) * torch.sqrt(
            torch.abs(best_bbox_preds[..., 2:4] + self.epsilon)
        )
        gts_bboxes[..., 2:4] = torch.sqrt(gts_bboxes[..., 2:4])
        loss_bbox = self.lambda_coord * self.mse(
            torch.flatten(best_bbox_preds, end_dim=-2),
            torch.flatten(gts_bboxes, end_dim=-2)
        )

        # Compute object confidence loss for when there is object
        best_conf_preds = I_obj_ij * (
            ((1 - best_ious) * preds[..., 24:25] + best_ious * preds[..., 29:30])
        )
        gts_conf = I_obj_ij * gts[..., 24:25]
        loss_obj = self.mse(
            torch.flatten(best_conf_preds),
            torch.flatten(gts_conf)
        )

        # Compute object confidence loss for when there is no object
        I_noobj_ij = 1 - I_obj_ij
        loss_noobj = self.mse(
            torch.flatten(I_noobj_ij * preds[..., 24:25], start_dim=1),
            torch.flatten(I_noobj_ij * gts[..., 24:25], start_dim=1)
        )
        loss_noobj += self.mse(
            torch.flatten(I_noobj_ij * preds[..., 29:30], start_dim=1), 
            torch.flatten(I_noobj_ij * gts[..., 24:25], start_dim=1)
        )
        loss_noobj = self.lambda_noobj * loss_noobj

        # Compute class loss
        I_obj_i = I_obj_ij
        loss_class = self.mse(
            torch.flatten(I_obj_i * preds[..., :20], end_dim=-2),
            torch.flatten(I_obj_i * gts[..., :20], end_dim=-2)
        )

        # Compute total loss 
        total_loss = loss_bbox + loss_obj + loss_noobj + loss_class
        return total_loss
