# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        assert cost_class != 0 or cost_bbox != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, pred_mask=None, gt_mask=None, max_thresh=1e6):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
            pred_mask: (B x num_target_boxes) bool matrix, where False indicates the positions in outputs to be masked out.
            gt_mask: (B x num_target_boxes) bool matrix, where False indicates the positions in targets to be masked out.
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_x1y1x2y2 = outputs['x1y1x2y2'].flatten(0, 1)  # [batch_size * num_queries, n_view, 3]
        pred_mask = pred_mask.flatten(0, 1)

        # Also concat the target labels and boxes
        if gt_mask is None:
            gt_mask = torch.ones(size=targets['x1y1x2y2'].shape[:3], dtype=torch.bool, device=out_prob.device)

        gt_obj_mask = gt_mask.sum(dim=-1).bool()
        gt_obj_mask = gt_obj_mask.flatten(0, 1)

        tgt_ids = targets['cls'].flatten(0, 1)
        tgt_x1y1x2y2 = targets['x1y1x2y2'].flatten(0, 1)
        view_mask = gt_mask.flatten(0, 1)
        sizes = targets['cls'].shape[0] * [targets['cls'].shape[1]]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        b1, b2 = out_x1y1x2y2.size(0), tgt_x1y1x2y2.size(0)
        out_x1y1x2y2 = out_x1y1x2y2[:, None].expand(-1, b2, -1, -1).flatten(0, 1)
        tgt_x1y1x2y2 = tgt_x1y1x2y2[None].expand(b1, -1, -1, -1).flatten(0, 1)
        cost_bbox = torch.abs(out_x1y1x2y2 - tgt_x1y1x2y2).sum(dim=-1)
        cost_bbox = cost_bbox.view(b1, b2, -1)
        cost_bbox = cost_bbox * view_mask[None]
        cost_bbox = cost_bbox.sum(dim=-1) / (view_mask.sum(dim=-1)[None] + 1e-6)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
        C[:, torch.logical_not(gt_obj_mask)] = max_thresh
        C[torch.logical_not(pred_mask)] = max_thresh
        C = C.view(bs, num_queries, -1).cpu()
        C_batch = C.split(sizes, -1)

        indices = []
        for batch_id, c in enumerate(C_batch):
            c_per_batch = c[batch_id]
            pred_ids, gt_ids = linear_sum_assignment(c_per_batch)
            valid_pairs = c_per_batch[pred_ids, gt_ids] < max_thresh
            valid_pairs = valid_pairs.cpu().numpy()
            indices.append((pred_ids[valid_pairs], gt_ids[valid_pairs]))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def get_area(x1y1x2y2):
    return torch.prod((x1y1x2y2[..., 2:4] - x1y1x2y2[..., :2]), dim=-1)

def box_iou(boxes1, boxes2):
    area1 = get_area(boxes1)
    area2 = get_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, 0] * wh[:, 1]  # [N,M]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, 0] * wh[:, 1]

    return iou - (area - union) / area