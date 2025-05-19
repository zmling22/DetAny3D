import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml
import types

from detect_anything.datasets.utils import *
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import open3d as o3d
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import math
import argparse

from scipy.spatial import ConvexHull
import colorsys

def adjust_brightness(color, factor=1.5, v_min=0.3):
    """在 HSV 空间调整亮度，避免过暗"""
    r, g, b = color
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    v = max(v, v_min) * factor  # 强制最低亮度 + 增强
    v = min(v, 1.0)  # 防止过曝
    return colorsys.hsv_to_rgb(h, s, v)

def save_point_cloud(depth_map, images, intrinsic_for_vis, filename, h, w):
    """
    保存点云文件。
    
    参数:
    - depth_map: 深度图数据。
    - origin_img: 用于得到点云上色的彩色图像。
    - intrinsic_for_vis: 相机内参矩阵。
    - filename: 保存点云的文件名。
    - h, w: 图像的高度和宽度。
    """
    # 计算点云的三维坐标
    origin_img = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1) * images[0].squeeze(0).detach().cpu() + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    color_image = origin_img.clone()[:, :h, :w].permute(1,2,0)
    color_image = Image.fromarray(color_image.cpu().numpy().astype('uint8'), 'RGB')

    pred = depth_map.detach().cpu().numpy()
    FX = intrinsic_for_vis[0, 0, 0].cpu().numpy()
    FY = intrinsic_for_vis[0, 1, 1].cpu().numpy()
    BX = intrinsic_for_vis[0, 0, 2].cpu().numpy()
    BY = intrinsic_for_vis[0, 1, 2].cpu().numpy()
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = (x - BX) / FX
    y = (y - BY) / FY
    z = np.array(pred)
    
    # 生成点云的点坐标
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    
    # 获取彩色图像作为点云的颜色
    colors = np.array(color_image).reshape(-1, 3) / 255.0
    
    # 创建并保存点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)

def save_color_image(images, h, w, filename):
    """
    保存原始的彩色图像。
    
    参数:
    - data: 原始的图像数据。
    - h, w: 图像的高度和宽度。
    - filename: 保存彩色图像的文件名。
    """
    # 恢复经过归一化处理的彩色图像
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    origin_img = pixel_std * images[0].squeeze(0).detach().cpu() + pixel_mean
    
    # 裁剪到指定大小并转换为图像格式
    color_image = origin_img[:, :h, :w].permute(1, 2, 0)
    color_image = Image.fromarray(color_image.cpu().numpy().astype('uint8'), 'RGB')
    
    # 保存彩色图像
    color_image.save(filename)
def save_depth_image(depth_map, filename, max_depth=None):
    """
    保存深度图为彩色图像。
    
    参数:
    - depth_map: 深度图数据。
    - filename: 保存深度图的文件名。
    - max_depth: 可选，深度图的最大深度值，用于归一化显示。
    """
    # 克隆深度图并对其进行归一化处理
    d = depth_map.clone()

    if max_depth is None:
        max_depth = int(d.max())  # 如果未指定最大深度，则使用深度图中的最大值
    d = colorize(d[0], 0, max_depth)
    # 保存彩色深度图
    Image.fromarray(d).save(filename)
def init_metrics(device_id):
    """
    初始化所有评估时需要用到的 metrics。
    这些 metrics 将被存储在一个字典中，并分配到指定设备上。
    """
    scene_level_metrics = {
        'delta_05': torch.zeros(1).to(device_id),
        'delta_1': torch.zeros(1).to(device_id),
        'delta_2': torch.zeros(1).to(device_id),
        'delta_3': torch.zeros(1).to(device_id),
        'abs_rel': torch.zeros(1).to(device_id),
        'rms': torch.zeros(1).to(device_id),
        'rms_log': torch.zeros(1).to(device_id),
        'si_log': torch.zeros(1).to(device_id),
        'log_10': torch.zeros(1).to(device_id),
        'error_fx_all': torch.zeros(1).to(device_id),
        'error_fy_all': torch.zeros(1).to(device_id),
        'error_bx_all': torch.zeros(1).to(device_id),
        'error_by_all': torch.zeros(1).to(device_id),
    }
    object_level_metrics = {
        # stage 2 metrics
        'giou_2d': torch.zeros(1).to(device_id),
        'iou_3d': torch.zeros(1).to(device_id),
        'acc_07': torch.zeros(1).to(device_id),
        'acc_05': torch.zeros(1).to(device_id),
        'acc_025': torch.zeros(1).to(device_id),
        'acc_015': torch.zeros(1).to(device_id),
    }
    return scene_level_metrics, object_level_metrics

def compute_metrics(scene_level_metrics, 
                    object_level_metrics, 
                    depth_gt, 
                    depth_map, 
                    masks, 
                    intrinsic_pred, 
                    intrinsic_gt, 
                    image_h, 
                    image_w,
                    decoded_bboxes_pred_2d = None,
                    gt_bboxes_2d = None,
                    decoded_bboxes_pred_3d = None,
                    gt_bboxes_3d = None,
                    rot_mat = None,
                    gt_rot_mat = None):

    depth_map = depth_map[masks > 0]
    depth_gt = depth_gt[masks > 0]
    if depth_gt.numel() > 0:
        scene_level_metrics['delta_05'] += ((torch.max(depth_gt / depth_map, depth_map / depth_gt) < 1.25 ** 0.5).float().mean())
        scene_level_metrics['delta_1'] += ((torch.max(depth_gt / depth_map, depth_map / depth_gt) < 1.25).float().mean())
        scene_level_metrics['delta_2'] += ((torch.max(depth_gt / depth_map, depth_map / depth_gt) < 1.25 ** 2).float().mean())
        scene_level_metrics['delta_3'] += ((torch.max(depth_gt / depth_map, depth_map / depth_gt) < 1.25 ** 3).float().mean())
        scene_level_metrics['abs_rel'] += (torch.abs(depth_gt - depth_map) / depth_gt).mean()
        scene_level_metrics['rms'] += torch.sqrt(((depth_gt - depth_map) ** 2).mean())
        scene_level_metrics['rms_log'] += torch.sqrt(((torch.log(depth_gt) - torch.log(depth_map)) ** 2).mean())
        scene_level_metrics['si_log'] += 100 * torch.std(torch.log(depth_map) - torch.log(depth_gt))
        scene_level_metrics['log_10'] += (torch.abs(torch.log10(depth_gt) - torch.log10(depth_map))).mean()

    error_fx, error_fy, error_f, error_bx, error_by, error_b = compute_intrinsic_measure(
                            Kest=torch.clone(intrinsic_pred),
                            Kgt=torch.clone(intrinsic_gt),
                            h=image_h,
                            w=image_w,
                        )

    scene_level_metrics['error_bx_all'] += error_bx
    scene_level_metrics['error_by_all'] += error_by
    scene_level_metrics['error_fx_all'] += error_fx
    scene_level_metrics['error_fy_all'] += error_fy

    if gt_bboxes_2d is not None:
        object_level_metrics['giou_2d'] += bbox_overlaps_giou(box_cxcywh_to_xyxy(decoded_bboxes_pred_2d), gt_bboxes_2d).detach().sum()
        for obj_i in range(gt_bboxes_3d.shape[0]):
            x, y, z, w, h, l, yaw = gt_bboxes_3d[obj_i].detach().cpu().numpy()
            # if gt_rot_mat is not None:
            #     pose = gt_rot_mat[obj_i]
            #     yaw = math.atan2(pose[0, 0], pose[2, 0]) + np.pi / 2 * 3
            vertices_3d_gt, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw)#, gt_rot_mat[obj_i].cpu().numpy())
            
            x, y, z, w, h, l, yaw = decoded_bboxes_pred_3d[obj_i].detach().cpu().numpy()
            # if rot_mat is not None:
            #     pose = rot_mat[obj_i]
            #     yaw = math.atan2(pose[0, 0], pose[2, 0]) + np.pi / 2 * 3
            vertices_3d_pred, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw)#, rot_mat[obj_i].cpu().numpy())
            iou_3d_i = box3d_iou(vertices_3d_gt, vertices_3d_pred)
            if iou_3d_i >= 0.7:
                object_level_metrics['acc_07'] += 1
            if iou_3d_i >= 0.5:
                object_level_metrics['acc_05'] += 1
            if iou_3d_i >= 0.25:
                object_level_metrics['acc_025'] += 1
            if iou_3d_i >= 0.15:
                object_level_metrics['acc_015'] += 1
            object_level_metrics['iou_3d'] += iou_3d_i

def collector(data):
    ret_dict = {}
    for k in data[0].keys():
        if k == 'prepare_for_dsam' or k == 'img_path':
            ret_dict[k] = data[0][k]
        else:
            k_list = list()
            for unit in data:
                k_list.append(unit[k])
            k_for_batch = torch.stack(k_list)
            ret_dict[k] = k_for_batch
    return ret_dict

def encode_gts(target_dict, cfg, device_id):
    
    encoded_gt_bboxes_2d = target_dict['gt_bboxes_2d'] / cfg.model.pad

    gt_alpha = -torch.atan2(
        box_xyxy_to_cxcywh(target_dict['gt_bboxes_2d'])[..., 0] - target_dict['gt_K'][..., 0, 2].to(device_id), target_dict['gt_K'][..., 0, 0].to(device_id)) + target_dict['gt_bboxes_3d'][..., 6]
    gt_alpha[gt_alpha > torch.pi] = gt_alpha[gt_alpha > torch.pi] - 2 * torch.pi
    gt_alpha[gt_alpha < -torch.pi] = gt_alpha[gt_alpha < -torch.pi] + 2 * torch.pi

    gt_angle_cls, gt_angle_res = angle2class(gt_alpha)
    target_dict['gt_angle_cls'] = gt_angle_cls
    target_dict['gt_angle_res'] = gt_angle_res

def preprocess(data, cfg, device_id):
    input_dict = {}
    target_dict = {}
    input_dict['images'] = data['images']
    input_dict['images_shape'] = data['before_pad_size']
    input_dict['vit_pad_size'] = data['vit_pad_size']
    target_dict['gt_K'] = data['K']

    if cfg.provide_gt_intrinsics:
        input_dict['gt_intrinsic'] = data['K']
    
    if cfg.merge_dino_feature:
        input_dict["image_for_dino"] = data["image_for_dino"].to(device_id)

    if cfg.tune_with_depth:
        depth_gt = data['depth'][:, :image_h, :image_w].to(device_id)
        masks = data['masks'][:, :image_h, :image_w].to(device_id)
        gt_angles = generate_rays(data['K'].to(device_id), (image_h, image_w))[1]
        target_dict['depth_gt'] = depth_gt
        target_dict['gt_angles'] = gt_angles
        target_dict['masks'] = masks

    if cfg.tune_with_prompt:
        # fetch gt here
        gt_center_2d = torch.stack([data['prepare_for_dsam'][i]['center_2d'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
        gt_bboxes_2d = torch.stack([data['prepare_for_dsam'][i]['bbox_2d'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
        gt_bboxes_3d = torch.stack([data['prepare_for_dsam'][i]['bbox_3d'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
        target_dict['gt_bboxes_2d'] = gt_bboxes_2d
        target_dict['gt_bboxes_3d'] = gt_bboxes_3d
        target_dict['gt_center_2d'] = gt_center_2d
        
        if cfg.input_point_prompt:
            input_dict['point_coords'] = torch.stack([data['prepare_for_dsam'][i]['point_coords'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
        if cfg.input_box_prompt:
            input_dict['boxes_coords'] = gt_bboxes_2d

        if cfg.input_point_prompt and cfg.input_box_prompt and cfg.random_drop_prompt:
            if random.random() < 0.8:
                del input_dict['boxes_coords']

        gt_pose = None
        if cfg.output_rotation_matrix:
            gt_pose = torch.stack([data['prepare_for_dsam'][i]['rotation_pose'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
        target_dict['gt_pose'] = gt_pose

        target_dict.update(encode_gts(target_dict, cfg))


def configure_opt_v3(cfg, model, train_loader):

    param_list = list()
    param_list_unidepth = list()

    param_list += list(model.module.sam.mask_decoder.parameters())
    if cfg.tune_with_depth:
        param_list_unidepth += list(model.module.sam.image_encoder.depth_head.parameters()) 
        param_list_unidepth += list(
            model.module.sam.image_encoder.dino.parameters()
        )
    param_list +=  list(model.module.sam.image_encoder.spm.parameters()) + list(model.module.sam.image_encoder.interactions.parameters()) \
        + list(model.module.sam.image_encoder.up.parameters())
    
    optimizer = torch.optim.AdamW(param_list, lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)

    optimizer.add_param_group({'params': model.module.sam.image_encoder.cls_token})
    optimizer.add_param_group({'params': model.module.sam.image_encoder.level_embed})
    
    optimizer.add_param_group({
        'params': param_list_unidepth,
        'lr': cfg.opt.unidepth_lr,  # 自定义学习率
        'weight_decay': cfg.opt.unidepth_weight_decay,  # 自定义权重衰减
    })

    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                                        T_max =  cfg.num_epochs)
    
    return optimizer, scheduler



def save_checkpoint(state, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)

def initcoords2D(b, h, w, device, homogeneous=False):
    """ Init Normalized Pixel Coordinate System
    """

    query_coords = torch.meshgrid(
        (
            torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
            torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
        ),
        indexing='ij'
    )
    query_coordsx, query_coordsy = query_coords[1], query_coords[0]

    if homogeneous:
        query_coords = torch.stack((query_coordsx, query_coordsy, torch.ones_like(query_coordsx)), dim=0).view([1, 3, h, w]).expand([b, 3, h, w])
    else:
        query_coords = torch.stack((query_coordsx, query_coordsy), dim=0).view([1, 2, h, w]).expand([b, 2, h, w])
    return query_coords

def get_sample_idx(h, w, prob=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if prob is not None:
        prob = prob.view([1, int(h * w)]).squeeze().cpu().numpy()
        sampled_index = np.random.choice(
            np.arange(int(h * w)),
            size=20000,
            replace=False,
            p=prob,
        )
    else:
        sampled_index = np.random.choice(
            np.arange(int(h * w)),
            size=20000,
            replace=False,
        )
    return sampled_index

def sample_wo_neighbour(x, sampled_index):
    assert len(x) == 1
    _, ch, h, w = x.shape
    x = x.contiguous().view([ch, int(h * w)])
    return x[:, sampled_index]

def minimal_solver(coords2Ds, normalrays, RANSAC_trial):
    """ RANSAC Minimal Solver
    """
    minimal_sample = 2
    device = coords2Ds.device

    sample_num = int(minimal_sample * RANSAC_trial)
    coords2Dc, normal = coords2Ds[:, 0:sample_num], normalrays[:, 0:sample_num]

    x1, y1, _ = torch.split(coords2Dc, 1, dim=0)
    n1, n2, n3 = torch.split(normal, 1, dim=0)

    n1 = n1 / n3
    n2 = n2 / n3

    x1, y1 = x1.view(minimal_sample, RANSAC_trial), y1.view(minimal_sample, RANSAC_trial)
    n1, n2 = n1.view(minimal_sample, RANSAC_trial), n2.view(minimal_sample, RANSAC_trial)

    fx = (x1[1] - x1[0]) / (n1[1] - n1[0] + 1e-10)
    bx = (x1[0] - n1[0] * fx) * 0.5 + (x1[1] - n1[1] * fx) * 0.5

    fy = (y1[1] - y1[0]) / (n2[1] - n2[0] + 1e-10)
    by = (y1[0] - n2[0] * fy) * 0.5 + (y1[1] - n2[1] * fy) * 0.5

    intrinsic = torch.eye(3).view([1, 3, 3]).repeat([len(fx), 1, 1]).to(device)
    intrinsic[:, 0, 0] = fx
    intrinsic[:, 1, 1] = fy
    intrinsic[:, 0, 2] = bx
    intrinsic[:, 1, 2] = by

    return intrinsic

def scoring_function_xy(normal_RANSAC, normal_ref):
    """ RANSAC Scoring Function
    """
    xx, yy, _ = torch.split(normal_RANSAC, 1, dim=1)
    xxref, yyref, zzref = torch.split(normal_ref, 1, dim=0)
    xxref = xxref / zzref
    yyref = yyref / zzref

    diffx = torch.sum((xx - xxref.unsqueeze(0)).abs() < 0.02, dim=[1, 2])
    diffy = torch.sum((yy - yyref.unsqueeze(0)).abs() < 0.02, dim=[1, 2])

    return diffx, diffy

def unnorm_intrinsic(intrinsic, b, h, w, device):
    """ Unmap Intrinsic to Image Coordinate System [0.5, h / w - 0.5]
    """
    scaleM = torch.eye(3).view([1, 3, 3]).expand([b, 3, 3]).to(device)
    scaleM[:, 0, 0] = float(1 / w) * 2
    scaleM[:, 1, 1] = float(1 / h) * 2

    scaleM[:, 0, 2] = -1.0
    scaleM[:, 1, 2] = -1.0
    return scaleM.inverse() @ intrinsic

def calibrate_camera_4DoF(incidence, RANSAC_trial=2048):
    """ 4DoF RANSAC Camera Calibration

    Args:
        incidence (tensor): Incidence Field
        RANSAC_trial (int): RANSAC Iteration Number. Default: 2048.
    """
    # Calibrate assume a simple pinhole camera model
    b, _, h, w = incidence.shape
    device = incidence.device
    coords2D = initcoords2D(b, h, w, device, homogeneous=True)

    sampled_index = get_sample_idx(h, w)
    normalrays = sample_wo_neighbour(incidence, sampled_index)
    coords2Ds = sample_wo_neighbour(coords2D, sampled_index)

    # Prepare for RANSAC
    intrinsic = minimal_solver(coords2Ds, normalrays, RANSAC_trial)

    valid = (intrinsic[:, 0, 0] < 1e-2).float() + (intrinsic[:, 1, 1] < 1e-2).float()
    valid = valid == 0
    intrinsic = intrinsic[valid]

    # RANSAC Loop
    intrinsic_inv = torch.linalg.inv(intrinsic)
    normalray_ransac = intrinsic_inv @ coords2Ds.unsqueeze(0)
    diffx, diffy = scoring_function_xy(normalray_ransac, normalrays)
    intrinsic_x, intrinsic_y = intrinsic, intrinsic

    maxid = torch.argmax(diffx)
    fx, bx = intrinsic_x[maxid, 0, 0], intrinsic_x[maxid, 0, 2]
    maxid = torch.argmax(diffy)
    fy, by = intrinsic_y[maxid, 1, 1], intrinsic_y[maxid, 1, 2]

    intrinsic_opt = torch.eye(3).to(device)
    intrinsic_opt[0, 0] = fx
    intrinsic_opt[0, 2] = bx
    intrinsic_opt[1, 1] = fy
    intrinsic_opt[1, 2] = by

    intrinsic_opt = unnorm_intrinsic(intrinsic_opt.unsqueeze(0), b, h, w, device)
    return intrinsic_opt.squeeze(0)

def compute_intrinsic_measure(Kest, Kgt, h, w):
    fxest, fyest, bxest, byest = Kest[0, 0], Kest[1, 1], Kest[0, 2], Kest[1, 2]
    fxgt, fygt, bxgt, bygt = Kgt[0, 0], Kgt[1, 1], Kgt[0, 2], Kgt[1, 2]

    error_fx = ((fxest - fxgt) / fxgt).abs().item()
    error_fy = ((fyest - fygt) / fygt).abs().item()

    error_f = max(
        error_fx,
        error_fy,
    )

    error_bx = (bxest - bxgt).abs().item() / w * 2
    error_by = (byest - bygt).abs().item() / h * 2
    error_b = max(
        error_bx,
        error_by
    )

    return error_fx, error_fy, error_f, error_bx, error_by, error_b

def grad(x):
    # x.shape : n, c, h, w
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    mag = diff_x**2 + diff_y**2
    # angle_ratio
    angle = torch.atan(diff_y / (diff_x + 1e-10))
    return mag, angle

def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]

class GradL1Loss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(GradL1Loss, self).__init__()
        self.name = 'GradL1'

    def forward(self, input, target, mask=None):

        grad_gt = grad(target)
        grad_pred = grad(input)
        mask_g = grad_mask(mask)

        loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
        loss = loss + \
            nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])

        return loss

class AffineInvarientLoss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(AffineInvarientLoss, self).__init__()
        self.name = 'AffineInvarient'

    def forward(self, input, target, mask=None):
        t_input = input.median()
        t_target = target.median()
        s_input = torch.abs(input - t_input).mean()
        s_target = torch.abs(target - t_target).mean()

        input_bar = (input - t_input) / s_input
        target_bar = (target - t_target) / s_target

        loss = torch.abs(input_bar - target_bar).mean()
        return loss

class GradL2Loss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(GradL2Loss, self).__init__()
        self.name = 'GradL2'

    def forward(self, input, target, mask=None):

        grad_gt = grad(target)
        grad_pred = grad(input)
        mask_g = grad_mask(mask)

        loss = nn.functional.mse_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
        loss = loss + \
            nn.functional.mse_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])

        return loss

def SILogLoss(input, target, coefficient, masks = None, log_mode = False):

    if masks is not None:
        input = input[masks > 0]
        target = target[masks > 0]
    if input.numel() == 0:
        return None
    if log_mode:
        alpha = 1e-7
        g = torch.log(input + alpha) - torch.log(target + alpha)
    else:
        g = input - target

    Dg = torch.pow(g, 2).mean() - coefficient * torch.pow(torch.mean(g), 2)
    loss = torch.sqrt(Dg)

    if torch.isnan(loss):
        print("Nan SILog loss")
        print("Input min max", torch.min(input), torch.max(input))
    return loss

def save_mask_images(pred_masks, iou_predictions, image_h, image_w, gt_bboxes_2d, images, instance_ids, save_root, file_name = None, debug_mode = False):
    """Save the predicted masks as images."""
    origin_img = torch.Tensor(
            [58.395, 57.12, 57.375]).view(-1, 1, 1) * images[0, :, :image_h, :image_w].squeeze(0).detach().cpu() + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    origin_img = cv2.cvtColor(origin_img.permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR)
    pred_masks = pred_masks[..., :image_h, :image_w]
    
    for i in range(pred_masks.shape[0]):
        if file_name == None:
            to_save_file_name = f'{instance_ids[i]}'
        if debug_mode:  
            [bbox_x1, bbox_y1, bbox_x2, bbox_y2] = gt_bboxes_2d[i]
            coor = [(int(bbox_x1), int(bbox_y1)), (int(bbox_x2), int(bbox_y2))]
            for j in range(pred_masks[i].shape[0]):  # Assuming three channels to save
                to_draw_masks = pred_masks[i][j].clone() > 0
                to_draw_masks = to_draw_masks.detach().cpu().numpy()
                to_draw_image = origin_img.copy() * 0.5 + 255 * np.repeat(to_draw_masks[:, :, np.newaxis], 3, axis=-1)
                cv2.rectangle(to_draw_image, coor[0], coor[1], (0, 0, 255), 2)
                cv2.putText(to_draw_image, f'iou_score: {int(iou_predictions[i][j] * 1000) / 1000}', (200, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
                cv2.imwrite(f'{save_root}/{to_save_file_name}_{i}_{j}.jpg', to_draw_image)
        else:
            to_draw_masks = pred_masks[i][-1].clone() > 0.8
            to_draw_masks = 255 * to_draw_masks.detach().cpu().numpy()
            cv2.imwrite(f'{save_root}/{to_save_file_name}_mask.jpg', to_draw_masks)

def save_predicted_point_prompts(point_prompts, images, image_h, image_w):
    origin_img = torch.Tensor(
        [58.395, 57.12, 57.375]).view(-1, 1, 1) * images[0, :, :image_h, :image_w].squeeze(0).detach().cpu() + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    todo = cv2.cvtColor(origin_img.permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR)
    for i in range(len(point_prompts)):
        cv2.circle(todo, (int(point_prompts[i, 1]), int(point_prompts[i, 0])), 2, (0, 0, 255), 4)
    cv2.imwrite(f'check_point_prompt.jpg', todo)

def save_multi_level_pred_bbox(cfg, images, point_coords, 
                               K, gt_K, image_h, image_w, 
                               gt_bboxes, gt_bboxes_3d, gt_center_2d, gt_rot_mat, 
                               dataset_name, epoch, iter, 
                               pred_bbox_2d_tensor_all, 
                               pred_bbox_3d_depth_tensor_all,
                               pred_bbox_3d_dims_tensor_all,
                               pred_bbox_3d_alpha_cls_tensor_all,
                               pred_pose_6d_tensor_all,
                               pred_box_ious):

    # import ipdb;ipdb.set_trace()
    for j in range(cfg.model.multi_level_box_output):
        # if cfg.contain_edge_obj:
        #     pred_bbox_2d = pred_bbox_2d_tensor_all[j][..., 2:].sigmoid()
        # else:
        pred_bbox_2d = pred_bbox_2d_tensor_all[j][..., 2:]

        pred_center_2d = pred_bbox_2d_tensor_all[j][..., :2]

        pred_bbox_3d_depth = pred_bbox_3d_depth_tensor_all[j][..., :1]
        pred_bbox_3d_depth_log_variance = pred_bbox_3d_depth_tensor_all[j][..., 1:]
        
        pred_bbox_3d_alpha_cls = pred_bbox_3d_alpha_cls_tensor_all[j][..., :12]
        pred_bbox_3d_alpha_res = pred_bbox_3d_alpha_cls_tensor_all[j][..., 12:]

        pred_bbox_3d_dims = pred_bbox_3d_dims_tensor_all[j]

        if cfg.output_rotation_matrix:
            pred_pose = rotation_6d_to_matrix(pred_pose_6d_tensor_all[j])
        else:
            pred_pose = None

        todo_dict = {
            'pred_bbox_2d': pred_bbox_2d,
            'pred_bbox_3d_alpha_cls': pred_bbox_3d_alpha_cls,
            'pred_bbox_3d_alpha_res': pred_bbox_3d_alpha_res,
            'pred_center_2d': pred_center_2d,
            'pred_bbox_3d_depth': pred_bbox_3d_depth,
            'pred_bbox_3d_dims': pred_bbox_3d_dims,
            'pred_pose': pred_pose,

        }

        decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes(todo_dict, cfg, K)

        for i in range(point_coords.shape[0]):
            [bbox_x1, bbox_y1, bbox_x2, bbox_y2] = gt_bboxes[i]
            coor = [(int(bbox_x1), int(bbox_y1)), (int(bbox_x2), int(bbox_y2))]
            origin_img = torch.Tensor(
                [58.395, 57.12, 57.375]).view(-1, 1, 1) * images[0, :, :image_h, :image_w].squeeze(0).detach().cpu() + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
            todo = cv2.cvtColor(origin_img.permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR)
            cv2.circle(todo, (int(point_coords[i, 0]), int(point_coords[i, 1])), 2, (0, 0, 255), 4)
            cv2.rectangle(todo, coor[0], coor[1], (0, 0, 255), 2)

            [bbox_x1, bbox_y1, bbox_x2, bbox_y2] = box_cxcywh_to_xyxy(decoded_bboxes_pred_2d[i])
            coor = [(int(bbox_x1), int(bbox_y1)), (int(bbox_x2 ), int(bbox_y2))]
            cv2.rectangle(todo, coor[0], coor[1], (255, 0, 0), 2)
            cv2.imwrite(f'{cfg.exp_dir}/{dataset_name}_vis_pic_{epoch}_{iter}_prompt_{i}_pred_head_{j}.jpg', todo)

            
            todo = cv2.cvtColor(origin_img.permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR)
            x, y, z, w, h, l, yaw = gt_bboxes_3d[i].detach().cpu().numpy()
            if gt_rot_mat is not None:
                gt_rot_mat_i = gt_rot_mat[i].cpu().numpy()
            else:
                gt_rot_mat_i = None
            vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, gt_rot_mat_i)
            
            vertices_2d = project_to_image(vertices_3d, gt_K.squeeze(0))
            fore_plane_center_2d = project_to_image(fore_plane_center_3d, gt_K.squeeze(0))
            draw_bbox_2d(todo, vertices_2d, color=(0, 0, 255))
            cv2.circle(todo, fore_plane_center_2d[0].astype(int), 2, (0, 0, 255) , 2)
            # import ipdb;ipdb.set_trace()
            if pred_pose is not None:
                rot_mat_i = pred_pose[i].cpu().numpy()
            else:
                rot_mat_i = None
            x, y, z, w, h, l, yaw = decoded_bboxes_pred_3d[i].detach().cpu().numpy()
            vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rot_mat_i)
            vertices_2d = project_to_image(vertices_3d, K.cpu().squeeze(0))
            fore_plane_center_2d = project_to_image(fore_plane_center_3d, K.cpu().squeeze(0))
            draw_bbox_2d(todo, vertices_2d, color=(255, 0, 0))
            cv2.circle(todo, fore_plane_center_2d[0].astype(int), 2, (255, 0, 0) , 2)
            cv2.putText(todo, f'pred_iou_score: {int(pred_box_ious[i][j] * 1000) / 1000}', (200, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
            cv2.imwrite(f'{cfg.exp_dir}/{dataset_name}_vis_pic_{epoch}_{iter}_prompt_{i}_pred_head_{j}_3D.jpg', todo)

def draw_text(im, text, pos, scale=0.4, color='auto', font=cv2.FONT_HERSHEY_SIMPLEX, bg_color=(0, 255, 255),
              blend=0.33, lineType=1):

    text = str(text)
    pos = [int(pos[0]), int(pos[1])]

    if color == 'auto':
        
        if bg_color is not None:
            color = (0, 0, 0) if ((bg_color[0] + bg_color[1] + bg_color[2])/3) > 127.5 else (255, 255, 255)
        else:
            color = (0, 0, 0) 

    if bg_color is not None:

        text_size, _ = cv2.getTextSize(text, font, scale, lineType)
        x_s = int(np.clip(pos[0], a_min=0, a_max=im.shape[1]))
        x_e = int(np.clip(x_s + text_size[0] - 1 + 4, a_min=0, a_max=im.shape[1]))
        y_s = int(np.clip(pos[1] - text_size[1] - 2, a_min=0, a_max=im.shape[0]))
        y_e = int(np.clip(pos[1] + 1 - 2, a_min=0, a_max=im.shape[0]))

        im[y_s:y_e + 1, x_s:x_e + 1, 0] = im[y_s:y_e + 1, x_s:x_e + 1, 0]*blend + bg_color[0] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 1] = im[y_s:y_e + 1, x_s:x_e + 1, 1]*blend + bg_color[1] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 2] = im[y_s:y_e + 1, x_s:x_e + 1, 2]*blend + bg_color[2] * (1 - blend)
        
        pos[0] = int(np.clip(pos[0] + 2, a_min=0, a_max=im.shape[1]))
        pos[1] = int(np.clip(pos[1] - 2, a_min=0, a_max=im.shape[0]))

    cv2.putText(im, text, tuple(pos), font, scale, color, lineType)



def vis_prompt_func(cfg, images, point_coords, K, gt_K, image_h, image_w, pred_masks, gt_bboxes, bboxes_pred_2d, gt_bboxes_3d, bboxes_pred_3d, gt_center_2d, pred_center_2d, gt_rot_mat, rot_mat, dataset_name, epoch, iter):
    
    K = K.cpu().numpy()
    for i in range(point_coords.shape[0]):

        # visualize gt 2d bbox and point prompt
        [bbox_x1, bbox_y1, bbox_x2, bbox_y2] = gt_bboxes[i]
        coor = [(int(bbox_x1), int(bbox_y1)), (int(bbox_x2), int(bbox_y2))]

        origin_img = torch.Tensor(
            [58.395, 57.12, 57.375]).view(-1, 1, 1) * images[0, :, :image_h, :image_w].squeeze(0).detach().cpu() + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        todo = cv2.cvtColor(origin_img.permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR)
        cv2.circle(todo, (int(point_coords[i, 0]), int(point_coords[i, 1])), 2, (0, 0, 255), 4)
        cv2.circle(todo, (int(pred_center_2d[i, 0] * cfg.model.pad), int(pred_center_2d[i, 1] * cfg.model.pad)), 2, (255, 0, 0), 4)
        cv2.circle(todo, (int(gt_center_2d[i, 0]), int(gt_center_2d[i, 1])), 2, (255, 255, 0), 4)
        cv2.rectangle(todo, coor[0], coor[1], (0, 0, 255), 2)

        # visualize pred 2d bbox  
        [bbox_x1, bbox_y1, bbox_x2, bbox_y2] = box_cxcywh_to_xyxy(bboxes_pred_2d[i])
        coor = [(int(bbox_x1), int(bbox_y1)), (int(bbox_x2 ), int(bbox_y2))]
        cv2.rectangle(todo, coor[0], coor[1], (255, 0, 0), 2)
        cv2.imwrite(f'{cfg.exp_dir}/{dataset_name}_vis_pic_{epoch}_{iter}_{i}.jpg', todo)

        todo = cv2.cvtColor(origin_img.permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR)
        pixels = np.array(origin_img.permute(1, 2, 0).numpy()).reshape(-1, 3) / 255.0
        pixels = np.clip(pixels, 0, 1)
        # 改进点1：根据亮度加权采样（避免全随机选到过多暗色）
        brightness = pixels.mean(axis=1)  # 计算每个像素的亮度
        prob = brightness / brightness.sum()  # 亮度越高采样概率越大
        sampled_indices = np.random.choice(pixels.shape[0], 100, p=prob, replace=False)
        sampled_colors = pixels[sampled_indices]

        # 改进点2：按亮度排序而非直接排序
        sampled_colors = sorted(sampled_colors, key=lambda c: colorsys.rgb_to_hsv(*c)[2])

        # 应用亮度增强
        adjusted_colors = [adjust_brightness(c, factor=2.0, v_min=0.4) for c in sampled_colors]
        color = adjusted_colors[i]

        color = [min(255, c*255) for c in color]

        x, y, z, w, h, l, yaw = gt_bboxes_3d[i].detach().cpu().numpy()
        if gt_rot_mat is not None:
            gt_rot_mat_i = gt_rot_mat[i].cpu().numpy()
        else:
            gt_rot_mat_i = None
        vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, gt_rot_mat_i)
        
        vertices_2d = project_to_image(vertices_3d, gt_K.squeeze(0))
        fore_plane_center_2d = project_to_image(fore_plane_center_3d, gt_K.squeeze(0))
        draw_bbox_2d(todo, vertices_2d, color=color)
        cv2.circle(todo, fore_plane_center_2d[0].astype(int), 2, color , 2)
        if rot_mat is not None:
            rot_mat_i = rot_mat[i].cpu().numpy()
        else:
            rot_mat_i = None
        x, y, z, w, h, l, yaw = bboxes_pred_3d[i].detach().cpu().numpy()
        vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rot_mat_i)
        vertices_2d = project_to_image(vertices_3d, K.squeeze(0))
        fore_plane_center_2d = project_to_image(fore_plane_center_3d, K.squeeze(0))
        draw_bbox_2d(todo, vertices_2d, color=color)
        cv2.circle(todo, fore_plane_center_2d[0].astype(int), 2, color , 2)
        # import ipdb;ipdb.set_trace()
        crop_x_min = max(int(vertices_2d[:, 0].min() - 0.1*(vertices_2d[:, 0].max() - vertices_2d[:, 0].min())), 0)
        crop_x_max = min(int(vertices_2d[:, 0].max() + 0.1*(vertices_2d[:, 0].max() - vertices_2d[:, 0].min())), todo.shape[1])
        crop_y_min = max(int(vertices_2d[:, 1].min() - 0.1*(vertices_2d[:, 1].max() - vertices_2d[:, 1].min())), 0)
        crop_y_max = min(int(vertices_2d[:, 1].max() + 0.1*(vertices_2d[:, 1].max() - vertices_2d[:, 1].min())), todo.shape[0])


        if crop_x_max - crop_x_min <= 0 or crop_y_max - crop_y_min <= 0:
            continue
        # print(f'crop_x_min:{crop_x_min}, crop_x_max:{crop_x_max}, crop_y_min:{crop_y_min}, crop_y_max:{crop_y_max}')

        # cv2.imwrite(f'{cfg.exp_dir}/{dataset_name}_vis_pic_{epoch}_{iter}_{i}_3D.png', todo[crop_y_min:crop_y_max, crop_x_min:crop_x_max, ...])

        vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw)
        vertices_2d = project_to_image(vertices_3d, K.squeeze(0))
        fore_plane_center_2d = project_to_image(fore_plane_center_3d, K.squeeze(0))
        draw_bbox_2d(todo, vertices_2d, color=(255, 255, 255))
        cv2.circle(todo, fore_plane_center_2d[0].astype(int), 2, (255, 255, 255) , 2)
        cv2.imwrite(f'{cfg.exp_dir}/{dataset_name}_vis_pic_{epoch}_{iter}_{i}_3D.png', todo)

def vis_prompt_func_2(cfg, images, point_coords, K, gt_K, image_h, image_w, pred_masks, gt_bboxes, bboxes_pred_2d, gt_bboxes_3d, bboxes_pred_3d, gt_center_2d, pred_center_2d, gt_rot_mat, rot_mat, dataset_name, epoch, iter, pred_box_ious=None):
    
    # Prepare the intrinsic matrix
    K = K.cpu().numpy()
    
    # Prepare the image
    origin_img = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1) * images[0, :, :image_h, :image_w].squeeze(0).detach().cpu() + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    todo = cv2.cvtColor(origin_img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    
    # # Define a set of colors for the bounding boxes (more colors)
    # colors = [
    #     (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    #     (255, 0, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0), (128, 128, 0),
    #     (0, 128, 128), (128, 0, 0), (0, 255, 255), (255, 192, 203), (0, 0, 0),
    #     (255, 69, 0), (72, 61, 139), (255, 105, 180), (34, 139, 34), (255, 20, 147),
    #     (221, 160, 221), (0, 191, 255), (255, 215, 0), (139, 69, 19), (255, 99, 71),
    #     (0, 255, 127), (147, 112, 219), (72, 209, 204), (233, 150, 122), (255, 239, 0)
    # ]
    pixels = np.array(origin_img.permute(1, 2, 0).numpy()).reshape(-1, 3) / 255.0
    pixels = np.clip(pixels, 0, 1)
    # 改进点1：根据亮度加权采样（避免全随机选到过多暗色）
    brightness = pixels.mean(axis=1)  # 计算每个像素的亮度
    prob = brightness / brightness.sum()  # 亮度越高采样概率越大
    sampled_indices = np.random.choice(pixels.shape[0], 100, p=prob, replace=False)
    sampled_colors = pixels[sampled_indices]

    # 改进点2：按亮度排序而非直接排序
    sampled_colors = sorted(sampled_colors, key=lambda c: colorsys.rgb_to_hsv(*c)[2])

    # 应用亮度增强
    adjusted_colors = [adjust_brightness(c, factor=2.0, v_min=0.4) for c in sampled_colors]
    # Create additional blank space at the bottom of the image
    # bottom_space_height = 100  # Increase this value as needed
    # todo = cv2.copyMakeBorder(todo, 0, bottom_space_height, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))  # Add white space at the bottom
    
    # Keep track of the current position for writing the box information
    # current_y = image_h + 30  # Adjust based on image height and desired spacing
    
    for i in range(point_coords.shape[0]):
        # Get 3D bounding box parameters
        x, y, z, w, h, l, yaw = bboxes_pred_3d[i].detach().cpu().numpy()
        if rot_mat is not None:
            rot_mat_i = rot_mat[i].cpu().numpy()
        else:
            rot_mat_i = None
        # Compute the 3D vertices and 2D projections
        vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rot_mat_i)
        vertices_2d = project_to_image(vertices_3d, K.squeeze(0))
        fore_plane_center_2d = project_to_image(fore_plane_center_3d, K.squeeze(0))
        
        # Choose the color for the current bounding box
        # color = colors[i % len(colors)]  # Cycle through colors
        color = adjusted_colors[i]

        color = [min(255, c*255) for c in color]
        
        # If IoU scores are provided, get the best IoU for the current object
        if pred_box_ious is not None:
            best_j = torch.argmax(pred_box_ious[i])  # Get the index of the best IoU box
            iou_score = pred_box_ious[i][best_j].item()  # Get the IoU score as a scalar
            
            # Only continue if IoU is above a certain threshold (e.g., 0.3)
            if iou_score < 0.05:
                continue
            
            # Display the IoU score at the top-right corner of the bounding box
            # cv2.putText(todo, f'IoU: {iou_score:.2f}', (int(vertices_2d[0, 0]), int(vertices_2d[0, 1]) - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw the 2D bounding box (predicted)
        draw_bbox_2d(todo, vertices_2d, color=color)
        
        # Add bounding box size (whl) text at the bottom of the image
        # bbox_info = f'{w:.2f}x{h:.2f}x{l:.2f}'  # Format the width, height, length as a string
        # cv2.putText(todo, bbox_info, (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # current_y += 25  # Move down for the next line of text
    
    # Save the image with bounding boxes and IoU annotations
    cv2.imwrite(f'{cfg.exp_dir}/{dataset_name}_vis_pic_{iter}_3D.png', todo)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
# 定义 1x1 卷积层用于调整 student feature 通道数
def cosine_similarity_loss(teacher_feature, student_feature):
    # 计算 Cosine 相似度
    cos_sim = F.cosine_similarity(teacher_feature, student_feature, dim=1)  # 在通道维度上计算相似度
    loss = 1 - cos_sim.mean()  # 平均损失
    return loss

def distill_features(teacher_features, student_features, loss_fn, linear_align, channel_align):
    distill_loss = 0.0
    # 遍历4组特征
    for i in range(len(teacher_features)):
        teacher_patch_feature = teacher_features[i][0]  # [1, 1369, 1024]
        teacher_class_token = teacher_features[i][1]  # [1, 1024]

        student_feature = student_features[i][2]  # [1, 1280, 53, 64]

        # 1. 对 teacher_patch_feature 进行 reshape 成 [1, 1024, 37, 37]
        patch_h_w = int(teacher_patch_feature.shape[1] ** 0.5)  # 1369 -> 37
        teacher_patch_feature = teacher_patch_feature.permute(0, 2, 1).reshape(1, 1024, patch_h_w, patch_h_w)  # [1, 1024, 37, 37]

        # 2. 调整 student_feature 尺寸与 teacher_patch_feature 对齐，插值到 (37, 37)
        student_feature_resized = F.interpolate(student_feature, size=(patch_h_w, patch_h_w), mode='bilinear', align_corners=True)  # [1, 1280, 37, 37]

        # 3. 使用 1x1 卷积对 student_feature 通道数进行对齐
        student_feature_aligned = channel_align(student_feature_resized)  # [1, 1024, 37, 37]

        # 4. 选择损失函数，支持MSE, KL散度或Cosine相似度
        if loss_fn == 'cos':
            feature_loss = cosine_similarity_loss(teacher_patch_feature, student_feature_aligned)
        elif loss_fn == 'mse':
            feature_loss = F.mse_loss(student_feature_aligned, teacher_patch_feature)
        elif loss_fn == 'kl':
            teacher_softmax = F.softmax(teacher_patch_feature, dim=1)
            student_softmax = F.softmax(student_feature_aligned, dim=1)
            feature_loss = F.kl_div(student_softmax.log(), teacher_softmax, reduction='batchmean')
        else:
            raise ValueError(f"Unknown loss function {loss_fn}")

        # # 5. 对 class token 的蒸馏，先将 student_feature 全局池化再与 teacher_class_token 对齐
        # student_global_feature = F.adaptive_avg_pool2d(student_feature, (1, 1)).view(1, -1)  # [1, 1280]
        # student_global_feature_aligned = linear_align(student_global_feature)  # 全连接层调整通道
        # class_token_loss = cosine_similarity_loss(student_global_feature_aligned, teacher_class_token)  # [1, 1024] vs [1, 1024]
        # # print('class_token_loss', class_token_loss.item())
        # 6. 累加损失
        distill_loss += feature_loss 
        
    return distill_loss

def distill_features0(teacher_features, student_features, loss_fn, channel_reduction):
    distill_loss = 0.0
    height_sam, width_sam = student_features.shape[1], student_features.shape[2]
    # 调整Depth Anything的特征图为与SAM特征图相同的尺寸
    depth_anything_resized = F.interpolate(teacher_features, size=(height_sam, width_sam), mode='bilinear', align_corners=True)
    depth_anything_resized = channel_reduction(depth_anything_resized).squeeze(0)
    # 使用L2损失来计算蒸馏损失
    if loss_fn == 'cos':
        feature_loss = cosine_similarity_loss(depth_anything_resized, student_features)
    elif loss_fn == 'mse':
        feature_loss = F.mse_loss(student_features, depth_anything_resized)
    distill_loss += feature_loss
    return distill_loss

def preprocess_image_for_depth_anything(image, input_size=518):
    transform = transforms.Compose([
                                    transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),  # 缩放到518x518
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
                                    ])
    return transform(image)

def depth_to_pointcloud_with_mask(depth_map, masks, intrinsic_pred):
    """
    使用 mask 将深度图转换为点云
    """
    bz, h, w = depth_map.shape
    depth_map = depth_map.cpu().numpy()
    masks = masks.cpu().numpy()
    FX = intrinsic_pred[0, 0].cpu().numpy()
    FY = intrinsic_pred[1, 1].cpu().numpy()
    BX = intrinsic_pred[0, 2].cpu().numpy()
    BY = intrinsic_pred[1, 2].cpu().numpy()

    # 生成图像坐标
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = (x - BX) / FX
    y = (y - BY) / FY
    
    # 仅保留 mask 匹配的部分
    valid_mask = masks > 0
    valid_mask = valid_mask.squeeze(0)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    z_valid = depth_map.squeeze(0)[valid_mask]

    # 生成对应的 3D 点云 (X, Y, Z)
    points = np.stack((x_valid * z_valid, y_valid * z_valid, z_valid), axis=-1)
    return points

def chamfer_loss(vals, target):
    B = vals.shape[0]
    xx = vals.view(B, 8, 1, 3)
    yy = target.view(B, 1, 8, 3)
    l1_dist = (xx - yy).abs().sum(-1)
    l1 = (l1_dist.min(1).values.mean(-1) + l1_dist.min(2).values.mean(-1))
    return l1

import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def chamfer_loss_one_to_one(vals, target):
    """
    计算一一对应的 Chamfer Loss（使用匈牙利匹配）

    Args:
        vals: (B, 8, 3) 预测的角点
        target: (B, 8, 3) 真实角点

    Returns:
        loss: (B,) Chamfer Loss
    """
    B = vals.shape[0]
    loss_list = []

    for i in range(B):
        # 计算 L1 距离矩阵 (8, 8)
        l1_dist = torch.cdist(vals[i], target[i], p=1)  # (8, 3) -> (8, 8) pairwise L1 distance

        # 使用匈牙利算法找到最优匹配
        row_ind, col_ind = linear_sum_assignment(l1_dist.cpu().detach().numpy())  # 转 numpy 进行匹配

        # 选取最优匹配的 L1 距离
        matched_dist = l1_dist[row_ind, col_ind].mean()  # 计算匹配点的平均损失
        loss_list.append(matched_dist)

    # 转换为 Tensor 并计算 batch 均值
    loss = torch.stack(loss_list).mean()
    return loss

def chamfer_loss2(vals, target):
    B = vals.shape[0]
    
    l1_dist = (xx - yy).abs().sum(-1)
    
    return l1

def calculate_pred_pose_for_cubercnn(pred_bbox_3d, pred_rot_mat, device_id):
    R_inv = torch.tensor(
        [[-0., -0., -1.],
        [ 0.,  1.,  0.],
        [ 1.,  0.,  0.]]).to(device_id).to(pred_bbox_3d.dtype)
    if pred_rot_mat is not None:
        pred_pose_for_cubercnn = torch.matmul(pred_rot_mat, R_inv)
    else:
        yaws = pred_bbox_3d[..., 6:].cpu().numpy()
        yaw_only_rot_mat = np.zeros((yaws.shape[0], 3, 3))

        # 遍历每个 yaw 值，计算旋转矩阵
        for i in range(yaws.shape[0]):
            angle = yaws[i, 0]
            rotation_matrix = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
            yaw_only_rot_mat[i] = rotation_matrix
        yaw_only_rot_mat = torch.tensor(yaw_only_rot_mat).to(device_id).to(pred_bbox_3d.dtype)
        pred_pose_for_cubercnn = torch.matmul(yaw_only_rot_mat, R_inv)
    return pred_pose_for_cubercnn

def decode_bboxes(ret_dict, cfg, K_for_convert):
    """
    Decode 2D and 3D bounding boxes from prediction results.

    Args:
        ret_dict (dict): The prediction dictionary containing keys:
            - 'pred_bbox_2d': Predicted 2D bounding boxes.
            - 'pred_bbox_3d_alpha_cls': Predicted alpha class.
            - 'pred_bbox_3d_alpha_res': Predicted alpha residual.
            - 'pred_center_2d': Predicted 2D center.
            - 'pred_bbox_3d_depth': Predicted 3D depth.
            - 'pred_bbox_3d_dims': Predicted 3D dimensions.
        cfg (dict): Configuration containing model settings.
        K_for_convert (Tensor): Camera intrinsic matrix for converting image points to 3D space.

    Returns:
        decoded_bboxes_pred_2d (Tensor): The decoded 2D bounding boxes.
        decoded_bboxes_pred_3d (Tensor): The decoded 3D bounding boxes.
    """
    # Decode 2D bounding boxes
    bboxes_pred_2d = ret_dict['pred_bbox_2d']
    decoded_bboxes_pred_2d = bboxes_pred_2d * cfg.model.pad

    # Decode 3D rotation (alpha angle)
    pred_alpha_cls = torch.argmax(ret_dict['pred_bbox_3d_alpha_cls'], dim=-1)
    pred_alpha_res = ret_dict['pred_bbox_3d_alpha_res'][np.arange(pred_alpha_cls.shape[0]), pred_alpha_cls]
    
    # Assuming class2angle function is already defined and accessible
    pred_alpha = class2angle(pred_alpha_cls, pred_alpha_res)

    # Compute predicted yaw (rotation around Y axis)
    bboxes_pred_2d_center_x = decoded_bboxes_pred_2d[..., 0]
    bboxes_pred_2d_center_y = decoded_bboxes_pred_2d[..., 1]
    pred_ry = torch.atan2(
        bboxes_pred_2d_center_x - K_for_convert[..., 0, 2], K_for_convert[..., 0, 0]) + pred_alpha
    pred_ry = (pred_ry + torch.pi) % (2 * torch.pi) - torch.pi  # Normalize the angle to [-pi, pi]
    pred_ry = pred_ry.unsqueeze(-1)

    # Decode 3D center with depth
    pred_centers_2d_with_depth = torch.cat([ret_dict['pred_center_2d'] * cfg.model.pad, ret_dict['pred_bbox_3d_depth'].exp()], dim=-1)
    
    # Assuming points_img2cam function is already defined and accessible
    pred_centers_3d = points_img2cam(pred_centers_2d_with_depth, K_for_convert[0])

    # Decode 3D bounding box dimensions
    decoded_bboxes_pred_3d = torch.cat([pred_centers_3d, ret_dict['pred_bbox_3d_dims'].exp(), pred_ry], dim=-1)

    return decoded_bboxes_pred_2d, decoded_bboxes_pred_3d

def decode_bboxes_virtual_to_real(ret_dict, cfg, K_gt, K_pred):
    """
    Decode 2D and 3D bounding boxes from prediction results.

    Args:
        ret_dict (dict): The prediction dictionary containing keys:
            - 'pred_bbox_2d': Predicted 2D bounding boxes.
            - 'pred_bbox_3d_alpha_cls': Predicted alpha class.
            - 'pred_bbox_3d_alpha_res': Predicted alpha residual.
            - 'pred_center_2d': Predicted 2D center.
            - 'pred_bbox_3d_depth': Predicted 3D depth.
            - 'pred_bbox_3d_dims': Predicted 3D dimensions.
        cfg (dict): Configuration containing model settings.
        K_for_convert (Tensor): Camera intrinsic matrix for converting image points to 3D space.

    Returns:
        decoded_bboxes_pred_2d (Tensor): The decoded 2D bounding boxes.
        decoded_bboxes_pred_3d (Tensor): The decoded 3D bounding boxes.
    """
    # Decode 2D bounding boxes
    bboxes_pred_2d = ret_dict['pred_bbox_2d']
    decoded_bboxes_pred_2d = bboxes_pred_2d * cfg.model.pad

    # Decode 3D rotation (alpha angle)
    pred_alpha_cls = torch.argmax(ret_dict['pred_bbox_3d_alpha_cls'], dim=-1)
    pred_alpha_res = ret_dict['pred_bbox_3d_alpha_res'][np.arange(pred_alpha_cls.shape[0]), pred_alpha_cls]
    
    # Assuming class2angle function is already defined and accessible
    pred_alpha = class2angle(pred_alpha_cls, pred_alpha_res)

    # Compute predicted yaw (rotation around Y axis)
    bboxes_pred_2d_center_x = decoded_bboxes_pred_2d[..., 0]
    bboxes_pred_2d_center_y = decoded_bboxes_pred_2d[..., 1]
    pred_ry = torch.atan2(
        bboxes_pred_2d_center_x - K_pred[..., 0, 2], K_pred[..., 0, 0]) + pred_alpha
    pred_ry = (pred_ry + torch.pi) % (2 * torch.pi) - torch.pi  # Normalize the angle to [-pi, pi]
    pred_ry = pred_ry.unsqueeze(-1)
    K_gt = K_gt.to(ret_dict['pred_bbox_3d_depth'].device)
    # import ipdb;ipdb.set_trace()
    real_depth = torch.sqrt((K_gt[..., 1, 1] * K_gt[..., 0, 0]) / (K_pred[..., 0, 0] * K_pred[..., 0, 0])) * ret_dict['pred_bbox_3d_depth'].exp()
    # Assuming points_img2cam function is already defined and accessible
    # real_depth_x = K_gt[..., 0, 0] / K_pred[..., 0, 0] * ret_dict['pred_bbox_3d_depth'].exp()
    # real_depth_y = K_gt[..., 1, 1] / K_pred[..., 1, 1] * ret_dict['pred_bbox_3d_depth'].exp()
    # # Decode 3D center with depth
    # pred_centers_2d_with_depth_x = torch.cat([ret_dict['pred_center_2d'] * cfg.model.pad, real_depth_x], dim=-1)
    # pred_centers_2d_with_depth_y = torch.cat([ret_dict['pred_center_2d'] * cfg.model.pad, real_depth_y], dim=-1)
    # pred_centers_2d_with_depth = torch.cat([pred_centers_2d_with_depth_x[..., :1], pred_centers_2d_with_depth_y[..., 1:2], real_depth], dim=-1)

    # Decode 3D center with depth
    pred_centers_2d_with_depth = torch.cat([ret_dict['pred_center_2d'] * cfg.model.pad, real_depth], dim=-1)
    # new_K = torch.tensor([[int(K_gt[0, 0, 0]), 0, K_pred[0, 0, 2].item()],
    #                   [0, int(K_gt[0, 1, 1]), K_pred[0, 1, 2].item()],
    #                   [0, 0, 1]]).to(ret_dict['pred_bbox_3d_depth'].device)
    pred_centers_3d = points_img2cam(pred_centers_2d_with_depth, K_gt[0])

    # Decode 3D bounding box dimensions
    decoded_bboxes_pred_3d = torch.cat([pred_centers_3d, ret_dict['pred_bbox_3d_dims'].exp(), pred_ry], dim=-1)

    return decoded_bboxes_pred_2d, decoded_bboxes_pred_3d

class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        # import ipdb;ipdb.set_trace()
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice