# Copyright (c) OpenMMLab. All rights reserved.
from .approx_max_iou_assigner import ApproxMaxIoUAssigner
from .ascend_assign_result import AscendAssignResult
from .ascend_max_iou_assigner import AscendMaxIoUAssigner
from .assign_result import AssignResult
from .atss_assigner import ATSSAssigner
from .base_assigner import BaseAssigner
from .center_region_assigner import CenterRegionAssigner
from .grid_assigner import GridAssigner
from .hungarian_assigner import HungarianAssigner
from .mask_hungarian_assigner import MaskHungarianAssigner
from .max_iou_assigner import MaxIoUAssigner
from .point_assigner import PointAssigner
from .region_assigner import RegionAssigner
from .sim_ota_assigner import SimOTAAssigner
from .task_aligned_assigner import TaskAlignedAssigner
from .uniform_assigner import UniformAssigner
from .ranking_assigner import RankingAssigner
from .hierarchical_assigner import HieAssigner
from .rf_assigner import RFAssigner
from .madet_assigner import MAAssigner

__all__ = [
    'BaseAssigner', 'MaxIoUAssigner', 'ApproxMaxIoUAssigner', 'AssignResult',
    'PointAssigner', 'ATSSAssigner', 'CenterRegionAssigner', 'GridAssigner',
    'HungarianAssigner', 'RegionAssigner', 'UniformAssigner', 'SimOTAAssigner',
    'TaskAlignedAssigner', 'MaskHungarianAssigner', 'AscendAssignResult',
    'AscendMaxIoUAssigner','RankingAssigner', 'HieAssigner','RFAssigner', 'MAAssigner'
]
