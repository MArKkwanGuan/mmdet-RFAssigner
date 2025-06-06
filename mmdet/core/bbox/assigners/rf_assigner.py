from operator import gt
from turtle import back
import torch
import random

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner



@BBOX_ASSIGNERS.register_module()
class RFAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='BboxOverlaps2D'), 
                 assign_metric='kl',
                 topk=[1,0,0],
                 ratio=0.9,
                 overlaps_thr=0.8):
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.assign_metric = assign_metric
        self.topk = topk
        self.overlaps_thr = overlaps_thr
        self.ratio = ratio

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        overlaps = self.iou_calculator(gt_bboxes, bboxes, mode=self.assign_metric)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        num_anchors = bboxes.size(0)
        num_gts = gt_bboxes.size(0)

        anchor_cx = (bboxes[...,0]+bboxes[...,2])/2
        anchor_cy = (bboxes[...,1]+bboxes[...,3])/2
        ext_gt_bboxes = gt_bboxes[:,None,:].expand(num_gts, num_anchors, 4)
        left = anchor_cx - ext_gt_bboxes[...,0]
        right = ext_gt_bboxes[..., 2] - anchor_cx
        top = anchor_cy - ext_gt_bboxes[..., 1]
        bottom = ext_gt_bboxes[..., 3] - anchor_cy

        bbox_targets = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        assign_result = self.assign_wrt_ranking(inside_gt_bbox_mask, overlaps, self.topk[0])
        if self.topk[1] != 0:
            bboxes2 = self.anchor_rescale(bboxes, self.ratio)
            overlaps2 = self.iou_calculator(gt_bboxes, bboxes2, mode=self.assign_metric)
            assign_result = self.assign_wrt_ranking(assign_result.gt_inds, overlaps2, self.topk[1])
        if self.topk[2] != 0:
            bboxes3 = self.anchor_rescale(bboxes2, self.ratio)
            overlaps3 = self.iou_calculator(gt_bboxes, bboxes3, mode=self.assign_metric)
            assign_result = self.assign_wrt_ranking(assign_result.gt_inds, overlaps3, self.topk[2])
            
        assign_result.gt_inds = torch.transpose(assign_result.gt_inds, 0, 1)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_ranking(self, assign_result, overlaps, k):

        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        mask1 = assign_result <= 0  
        mask2 = assign_result > 0
        
        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_gts, num_bboxes),
                                             0,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_gts, num_bboxes))
            assigned_gt_inds = overlaps.new_full((num_gts, num_bboxes),
                                        0,
                                        dtype=torch.long)
            return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=None)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.topk(k, dim=1, largest=True, sorted=True)

        backgroud_mask = (~((max_overlaps >= 0) & (max_overlaps < self.overlaps_thr))).repeat(num_gts, 1)

        for i in range(num_gts):
            for j in range(k):
                max_overlap_inds = overlaps[i,:] == gt_max_overlaps[i,j]
                assigned_gt_inds[i, max_overlap_inds] = 1

        assigned_gt_inds = (assigned_gt_inds * backgroud_mask * mask1) + (assign_result * mask2)
        # non_zero_count = torch.count_nonzero(assigned_gt_inds ^ assign_result)
        # print(non_zero_count)

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=None)
    
    def anchor_rescale(self, bboxes, ratio):
        center_x2 = (bboxes[..., 2] + bboxes[..., 0]) / 2
        center_y2 = (bboxes[..., 3] + bboxes[..., 1]) / 2
        w2 = bboxes[..., 2] - bboxes[..., 0]
        h2 = bboxes[..., 3] - bboxes[..., 1]
        bboxes[..., 0] = center_x2 - w2*ratio/2
        bboxes[..., 1] = center_y2 - h2*ratio/2
        bboxes[..., 2] = center_x2 + w2*ratio/2
        bboxes[..., 3] = center_y2 + h2*ratio/2
        
        return bboxes







 

    


        