from scipy.optimize import linear_sum_assignment
from segment_anything.datasets.utils import box3d_iou

import numpy as np

def compute_3d_iou(corners1, corners2):
    """
    corners1: (N, 8, 3)
    corners2: (M, 8, 3)
    returns: ious: (N, M)
    """
    N = corners1.shape[0]
    M = corners2.shape[0]
    ious = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        for j in range(M):
            iou_3d = box3d_iou(corners1[i], corners2[j])
            ious[i, j] = iou_3d

    return ious


def compute_agnostic_cost(pred_boxes, gt_boxes):
    """
    pred_boxes_corners: (N, 8, 3)
    gt_boxes_corners: (M, 8, 3)
    pred_boxes: (N, 7)  e.g. 3D box格式 [x, y, z, dx, dy, dz, heading]
    gt_boxes:   (M, 7)

    Return:
        cost_matrix: (N, M)
    """
    # 这里假设你有一个 compute_3d_iou 接口
    # iou_3d = compute_3d_iou(pred_boxes_corners, gt_boxes_corners)  # (N, M)
    # iou_cost = 1 - iou_3d
    
    # 你也可加 L1 距离
    center_dist  = torch.cdist(pred_boxes[:, :3], gt_boxes[:, :3], p=1)
    size_dist    = torch.cdist(pred_boxes[:, 3:6], gt_boxes[:, 3:6], p=1)
    heading_dist = torch.cdist(pred_boxes[:, 6:], gt_boxes[:, 6:], p=1)

    cost_matrix = center_dist + size_dist + heading_dist
    return cost_matrix

def hungarian_match_agnostic(pred_boxes, gt_boxes):
    """
    Args:
        pred_boxes: (N, D), e.g. [N, 7] for 3D
        gt_boxes:   (M, D)

    Return:
        matched_pred_idx: (K,)  # 与某个GT匹配的预测索引
        matched_gt_idx:   (K,)  # K = min(N, M)
        unmatched_pred_idx:  其余未匹配到GT的预测索引
    """
    cost_mat = compute_agnostic_cost(pred_boxes, gt_boxes)  # (N, M)
    cost_mat_np = cost_mat.detach().cpu().numpy()

    row_ind, col_ind = linear_sum_assignment(cost_mat_np)
    # row_ind: 与GT成功匹配的 pred index
    # col_ind: row_ind对应的 gt index

    matched_pred_idx = row_ind
    matched_gt_idx   = col_ind

    # 若 N > M，还有一些预测框 unmatched
    all_pred_idx = set(range(len(pred_boxes)))
    matched_pred_set = set(matched_pred_idx.tolist())
    unmatched_pred_idx = list(all_pred_idx - matched_pred_set)

    return matched_pred_idx, matched_gt_idx, unmatched_pred_idx
