from segment_anything.datasets.stage2_dataset import *
from segment_anything.datasets.stage1_dataset import *
from train_utils import *
from wrap_model import WrapModel
from  segment_anything.modeling.unidepth_utils import generate_rays
from PIL import Image
import cv2
import yaml
from box import Box
from tqdm import tqdm
import os
import argparse
import open3d as o3d
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from datetime import datetime
from box import Box
from tqdm import tqdm
import os
from torchvision import transforms
import argparse
from torch.nn.utils.rnn import pad_sequence
from segment_anything.modeling.unidepth_utils import generate_rays
from ema_torch import ExponentialMovingAverage
from torch.utils.tensorboard import SummaryWriter
import random
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext  # Python 3.7+ 提供的空上下文管理器

def parse_args():
    parser = argparse.ArgumentParser(description='set config path') 
    parser.add_argument('--config_path', default='/cpfs01/user/jianghaoran/detany3d/DetAny3D0827/segment_anything/configs/dlc_config.yaml', type=str, help='abosulute path of the config') 
    parser.add_argument('--resume', type=str, help='Path to resume checkpoint')
    parser.add_argument('--exp_dir', type=str, help='Path to save checkpoint')
    args = parser.parse_args() 
    print(f'args: {args}')
    return args

def train_one_epoch(
    cfg,
    model,
    device_id,
    optimizer,
    scheduler,
    train_dataloader,
    epoch,
    logger,
    teacher_model = None,
    linear_align = None,
    channel_align = None,
    channel_reduction = None,
):
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    train_dataloader.sampler.set_epoch(epoch)
    model.train()
    loss_epoch = torch.zeros(1).to(device_id)
    loss_dict_epoch = {}
    print(scheduler.get_last_lr())
    with tqdm(total=len(train_dataloader)) as t:
        for iter, data in enumerate(train_dataloader):
            # import ipdb;ipdb.set_trace()
            optimizer.zero_grad()
            # input_dict, target_dict = preprocess(data, cfg, device_id)
            with autocast(cfg.use_amp):
                input_dict = {}
                input_dict['images'] = data['images']
                input_dict['images_shape'] = data['before_pad_size']
                input_dict['vit_pad_size'] = data['vit_pad_size']
                image_h, image_w = int(data['before_pad_size'][0, 0]), int(data['before_pad_size'][0, 1])

                if cfg.tune_with_depth:
                    # fetch the gt data here
                    # bug exists if batch size is not 1.
                    depth_gt = data['depth'][:, :image_h, :image_w].to(device_id)
                    masks = data['masks'][:, :image_h, :image_w].to(device_id)
                    gt_angles = generate_rays(data['K'].to(device_id), (image_h, image_w))[1]
                    phi_gt, theta_gt = gt_angles[..., 0], gt_angles[..., 1]

                    if cfg.merge_dino_feature:
                        input_dict["image_for_dino"] = data["image_for_dino"].to(device_id)

                if cfg.tune_with_prompt and len(data['prepare_for_dsam']) > 0:
                    # fetch gt here
                    gt_bboxes_2d = torch.stack([data['prepare_for_dsam'][i]['bbox_2d'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                    gt_center_2d = torch.stack([data['prepare_for_dsam'][i]['center_2d'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                    gt_bboxes_3d = torch.stack([data['prepare_for_dsam'][i]['bbox_3d'].to(device_id) for i in range(len(data['prepare_for_dsam']))])

                    if cfg.input_point_prompt:
                        input_dict['point_coords'] = torch.stack([data['prepare_for_dsam'][i]['point_coords'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                    if cfg.input_box_prompt:
                        input_dict['boxes_coords'] = torch.stack([data['prepare_for_dsam'][i]['boxes_coords'].to(device_id) for i in range(len(data['prepare_for_dsam']))])

                    if cfg.input_point_prompt and cfg.input_box_prompt and cfg.random_drop_prompt:
                        if random.random() < cfg.random_drop_prompt_prob:
                            # key_to_drop = random.choice(['point_coords', 'boxes_coords'])
                            del input_dict['boxes_coords']
                        else:
                            del input_dict['point_coords']
                    if cfg.input_depth_prompt:
                        input_dict['depth_coords'] = torch.stack([data['prepare_for_dsam'][i]['depth_coords'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                        if random.random() < cfg.random_drop_depth_prompt_prob:
                            del input_dict['depth_coords']
                        # else:
                        #     del input_dict['depth_coords']
                    
                    if cfg.model.detector_2d_head:
                        gt_masks = torch.stack([data['prepare_for_dsam'][i]['obj_mask'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                        input_dict['gt_masks'] = gt_masks
                    
                    gt_pose = None
                    if cfg.output_rotation_matrix:
                        gt_pose = torch.stack([data['prepare_for_dsam'][i]['rotation_pose'].to(device_id) for i in range(len(data['prepare_for_dsam']))])

                    # encode gts
                    prompt_num = gt_bboxes_2d.shape[0]
                    encoded_gt_bboxes_2d = gt_bboxes_2d / cfg.model.pad
                    encoded_gt_center_2d = gt_center_2d / cfg.model.pad

                    gt_alpha = -torch.atan2(
                        box_xyxy_to_cxcywh(gt_bboxes_2d)[..., 0] - data['K'][..., 0, 2].to(device_id), data['K'][..., 0, 0].to(device_id)) + gt_bboxes_3d[..., 6]
                    gt_alpha[gt_alpha > torch.pi] = gt_alpha[gt_alpha > torch.pi] - 2 * torch.pi
                    gt_alpha[gt_alpha < -torch.pi] = gt_alpha[gt_alpha < -torch.pi] + 2 * torch.pi

                    gt_angle_cls, gt_angle_res = angle2class(gt_alpha)
                    gt_dir_cls_onehot = F.one_hot(gt_angle_cls.long(), 12).float()

                    encoded_gt_bbox_3d_dim = gt_bboxes_3d[..., 3:6].log()
                    encoded_gt_bbox_3d_depth = gt_bboxes_3d[..., 2:3].log()

                    if cfg.model.multi_level_box_output > 1:
                        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
                        input_dict['gt_angle_cls'] = gt_angle_cls
                    
                    instance_masks = None
                    if 'instance_mask' in data['prepare_for_dsam'][0].keys():
                        instance_masks = torch.stack([data['prepare_for_dsam'][i]['instance_mask'].to(device_id) for i in range(len(data['prepare_for_dsam']))])


                only_2d_mode = len(data['prepare_for_dsam']) > 0 and torch.all(gt_bboxes_3d == -1).item()
                if cfg.provide_gt_intrinsics and not only_2d_mode:
                    input_dict['gt_intrinsic'] = data['K']
                
                ret_dict = model(input_dict)
                
                # import ipdb;ipdb.set_trace()
                K_for_convert = ret_dict.get('gt_intrinsic', ret_dict['pred_K']).to(device_id)
                
                loss_dict = {}
                if cfg.tune_detector_2d:
                    loss_dict['detector_2d_loss'] = 5 * ret_dict['detector_2d_loss']

                if cfg.tune_with_depth:
                    pred_angles = generate_rays(ret_dict['pred_K'], (image_h, image_w))[1]
                    phi_pred, theta_pred = pred_angles[..., 0], pred_angles[..., 1]

                    loss_phi = SILogLoss(phi_pred, phi_gt, coefficient=cfg.loss.phi.coefficient)
                    loss_theta = SILogLoss(theta_pred, theta_gt, coefficient=cfg.loss.theta.coefficient)
                    loss_depth = SILogLoss(ret_dict['depth_maps'], depth_gt, coefficient=cfg.loss.depth.coefficient, masks=masks, log_mode=True)
                
                    if 'depth_loss' in cfg.loss.loss_list and loss_depth is not None:
                        loss_dict['depth_loss'] = cfg.loss.depth.depth_loss_weight * loss_depth
                    if 'intrinsic_loss' in cfg.loss.loss_list and not only_2d_mode:
                        loss_dict['loss_phi'] = cfg.loss.phi.phi_loss_weight * loss_phi
                        loss_dict['loss_theta'] = cfg.loss.theta.theta_loss_weight * loss_theta

                if cfg.tune_with_prompt and len(data['prepare_for_dsam']) > 0:

                    decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes(ret_dict, cfg, K_for_convert)
                    # if cfg.model.enable_dense_prompt:
                    #     import ipdb;ipdb.set_trace()
                    #     from segment_anything.modeling.matcher import hungarian_match_agnostic
                    # 2d giou loss
                    bboxes_pred_2d = ret_dict['pred_bbox_2d']
                    batch_iou_2d = bbox_overlaps_giou(box_cxcywh_to_xyxy(bboxes_pred_2d), encoded_gt_bboxes_2d)
                    loss_box_giou = (1 - batch_iou_2d)

                    # 2d bbox l1 loss
                    loss_l1_bbox = F.smooth_l1_loss(bboxes_pred_2d, box_xyxy_to_cxcywh(encoded_gt_bboxes_2d), reduction='none', beta=1.0 / 9.0)

                    # 3d bbox center 2d loss
                    pred_bbox_3d_center_2d = ret_dict['pred_center_2d']
                    bbox_3d_center_2d_loss = F.smooth_l1_loss(pred_bbox_3d_center_2d, encoded_gt_center_2d, reduction='none', beta=1.0 / 9.0)

                    # 3d bbox depth loss
                    pred_bbox_3d_depth = ret_dict['pred_bbox_3d_depth']
                    depth_log_variance = ret_dict['pred_bbox_3d_depth_log_variance']
                    bbox_3d_depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(pred_bbox_3d_depth - encoded_gt_bbox_3d_depth) + depth_log_variance

                    # 3d bbox dim loss
                    pred_bbox_3d_dim = ret_dict['pred_bbox_3d_dims']
                    bbox_3d_dim_loss = F.smooth_l1_loss(pred_bbox_3d_dim, encoded_gt_bbox_3d_dim, reduction='none', beta=1.0 / 9.0 )

                    bbox_3d_alpha_cls_loss = F.cross_entropy(ret_dict['pred_bbox_3d_alpha_cls'], gt_angle_cls, reduction='none')
                    pred_boxes_3d_alpha_res = torch.sum(ret_dict['pred_bbox_3d_alpha_res'] * gt_dir_cls_onehot, 1)
                    bbox_3d_alpha_res_loss = F.smooth_l1_loss(pred_boxes_3d_alpha_res, gt_angle_res, reduction='none', beta=1.0 / 9.0)
                    bbox_3d_alpha_loss = bbox_3d_alpha_cls_loss + bbox_3d_alpha_res_loss

                    if '2d_bbox_loss' in cfg.loss.loss_list:
                        loss_dict['loss_2d_bbox'] = 5 * loss_l1_bbox.sum() / prompt_num
                        loss_dict['loss_2d_giou'] = 2 * loss_box_giou.sum() / prompt_num
                    if '3d_bbox_loss' in cfg.loss.loss_list and not only_2d_mode:
                        loss_dict['loss_3d_center_2d'] = 10 * bbox_3d_center_2d_loss.sum() / prompt_num
                        loss_dict['loss_3d_depth'] = 1 * bbox_3d_depth_loss.sum() / prompt_num
                        loss_dict['loss_3d_dim'] = 1 * bbox_3d_dim_loss.sum() / prompt_num
                        loss_dict['loss_3d_alpha'] = 1 * bbox_3d_alpha_loss.sum() / prompt_num
                    if cfg.model.multi_level_box_output > 1 and not only_2d_mode and ret_dict.get('box_iou_loss', None) is not None:
                        loss_dict['box_iou_loss'] = 2 * ret_dict['box_iou_loss']
                    if instance_masks is not None and 'mask_loss' in cfg.loss.loss_list:
                        instance_masks = instance_masks[..., :image_h, :image_w]
                        pred_masks = ret_dict['masks'][..., :image_h, :image_w]
                        iou_predictions = ret_dict['iou_predictions']
                        
                        loss_focal = torch.tensor(0., device=device_id)
                        loss_dice = torch.tensor(0., device=device_id)
                        loss_mask_iou = torch.tensor(0., device=device_id)

                        for pred_mask, gt_mask, iou_prediction in zip(pred_masks, instance_masks, iou_predictions): 
                            
                            # compute batch_iou of pred_mask and gt_mask
                            pred_mask = (pred_mask >= 0.5).float() 
                            intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1,2))
                            union = torch.sum(pred_mask, dim=(1,2))
                            epsilon = 1e-7
                            batch_iou = (intersection / (union + epsilon)).unsqueeze(1)

                            # Find the index with the maximum IoU
                            max_iou_idx = torch.argmax(batch_iou, dim=0).squeeze()

                            # Select the pred_mask and gt_mask corresponding to the max IoU index
                            selected_pred_mask = pred_mask[max_iou_idx]
                            
                            selected_batch_iou = batch_iou[max_iou_idx].squeeze()
                            selected_iou_prediction = iou_prediction[max_iou_idx]

                            # Calculate losses
                            loss_focal += focal_loss(selected_pred_mask, gt_mask)
                            loss_dice += dice_loss(selected_pred_mask, gt_mask)
                            loss_mask_iou += F.mse_loss(selected_iou_prediction, selected_batch_iou, reduction='sum')

                        loss_dict['loss_mask_focal'] = 20 * loss_focal / prompt_num
                        loss_dict['loss_mask_dice'] = 1 * loss_dice / prompt_num
                        loss_dict['loss_mask_iou'] = 1 * loss_mask_iou / prompt_num

                    pred_pose = None
                    if cfg.output_rotation_matrix:
                        pred_pose = rotation_6d_to_matrix(ret_dict['pred_pose_6d'])

                        # bboxes_pred_2d_center_x = decoded_bboxes_pred_2d[..., 0]
                        # bboxes_pred_2d_center_y = decoded_bboxes_pred_2d[..., 1]

                        # bbox_gt_2d_center_x = box_xyxy_to_cxcywh(gt_bboxes_2d)[..., 0]
                        # bbox_gt_2d_center_y = box_xyxy_to_cxcywh(gt_bboxes_2d)[..., 1]

                        # cube_pose_allocentric = pred_pose
                        # pred_pose = R_from_allocentric(K_for_convert, pred_pose, u=bboxes_pred_2d_center_x, v=bboxes_pred_2d_center_y)
                        # gt_poses_allocentric = R_to_allocentric(data['K'].to(device_id), gt_pose, u=bbox_gt_2d_center_x, v=bbox_gt_2d_center_y)
                        # loss_pose = 1 - so3_relative_angle(cube_pose_allocentric, gt_poses_allocentric, eps=0.1)
                        # import ipdb;ipdb.set_trace()
                        
                        if 'so3_loss' in cfg.loss.loss_list and not only_2d_mode:
                            loss_pose = 1 - so3_relative_angle(pred_pose, gt_pose, eps=0.1)
                            loss_dict['pose_loss'] = loss_pose.sum() / prompt_num

                    if cfg.chamfer_loss_supervise_all:
                        decoded_bboxes_pred_3d_corners = compute_3d_bbox_vertices_batch(decoded_bboxes_pred_3d, pred_pose)
                    else:
                        decoded_bboxes_pred_3d_corners = compute_3d_bbox_vertices_batch(gt_bboxes_3d, pred_pose)
                    decoded_bboxes_gt_3d_corners = compute_3d_bbox_vertices_batch(gt_bboxes_3d, gt_pose)

                    if 'corners_loss' in cfg.loss.loss_list and not only_2d_mode:
                        
                        # get pred 3d corners (originally)
                        # change to use gt bbox 3d to computer corners, for easizer convergence of rotation matrix
                        
                        
                        # corners_chamfer_loss = chamfer_loss(decoded_bboxes_pred_3d_corners, decoded_bboxes_gt_3d_corners)
                        corners_loss = 5 * F.smooth_l1_loss(decoded_bboxes_pred_3d_corners, decoded_bboxes_gt_3d_corners, reduction='none', beta=1.0 / 9.0)
                        loss_dict['corners_loss'] = corners_loss.sum() / prompt_num
                    
                    if 'chamfer_loss' in cfg.loss.loss_list and not only_2d_mode:

                        corners_chamfer_loss = chamfer_loss(decoded_bboxes_pred_3d_corners, decoded_bboxes_gt_3d_corners)
                        loss_dict['corners_chamfer_loss'] = corners_chamfer_loss.sum() / prompt_num
            # print(f'loss_dict: {loss_dict}')
            # import ipdb;ipdb.set_trace()
            if len(loss_dict) > 0:
                loss_total = sum([loss_dict[k] for k in loss_dict.keys()])
            else:
                loss_total = next(model.parameters()).new_zeros(1, requires_grad=True).sum()
            loss_epoch += loss_total.detach()
    
            for key, value in loss_dict.items():
                if key not in loss_dict_epoch:
                    loss_dict_epoch[key] = value.detach()
                else:
                    loss_dict_epoch[key] += value.detach()

            cfg.scaler.scale(loss_total).backward()
            cfg.scaler.step(optimizer)
            cfg.scaler.update()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            
            if cfg.use_ema:
                cfg.ema.update()

            if cfg.writer and dist.get_rank() == 0:
                for loss_name, loss_value in loss_dict.items():
                    cfg.writer.add_scalar(f'Train/{loss_name}', loss_value.item(), epoch * len(train_dataloader) + iter)

            if device_id == 0:
                t.update(1)

    loss_epoch /= (iter + 1)
    dist.all_reduce(loss_epoch)
    # print(scheduler.get_last_lr())
    scheduler.step()

    for k, v in loss_dict_epoch.items():
        loss_dict_epoch[k] /= (iter + 1)
        dist.all_reduce(loss_dict_epoch[k])

    if cfg.rank == 0:
        logger.info({'Loss': loss_epoch / cfg.world_size})
        t.set_description(desc="Epoch %i"%epoch)
        t.set_postfix(steps=iter, loss=(loss_epoch / cfg.world_size).data.item())
        avg_loss_dict = {k: (v / cfg.world_size).item() for k, v in loss_dict_epoch.items()}
        logger.info({"Epoch {} Detailed Losses".format(epoch): avg_loss_dict})

def validate_one_epoch(
    cfg,
    model,
    device_id,
    dataset_name,
    val_dataloader,
    epoch,
    logger):
    
    omni3d_result = []

    model.eval()
    # here is the score initialization
    scene_level_metrics, object_level_metrics = init_metrics(device_id)
    # count bbox and scenes
    total_bboxes = torch.zeros(1).to(device_id)
    total_scenes = torch.zeros(1).to(device_id)
    
    val_dataloader.sampler.set_epoch(epoch)
    ema_context = cfg.ema.average_parameters() if cfg.use_ema else nullcontext()
    with ema_context:
        with tqdm(total=len(val_dataloader)) as t:
            for iter, data in enumerate(val_dataloader):
                total_scenes += 1
                input_dict = {}
                input_dict['images'] = data['images']
                input_dict['images_shape'] = data['before_pad_size']
                input_dict['vit_pad_size'] = data['vit_pad_size']
                image_h, image_w = int(data['before_pad_size'][0, 0]), int(data['before_pad_size'][0, 1])
                # import ipdb;ipdb.set_trace()
                if cfg.inference_with_prompt and len(data['prepare_for_dsam']) > 0:
                    gt_bboxes_2d = torch.stack([data['prepare_for_dsam'][i]['bbox_2d'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                    gt_bboxes_3d = torch.stack([data['prepare_for_dsam'][i]['bbox_3d'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                    gt_center_2d = torch.stack([data['prepare_for_dsam'][i]['center_2d'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                    if cfg.inference_with_point_prompt:
                        input_dict['point_coords'] = torch.stack([data['prepare_for_dsam'][i]['point_coords'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                        # if cfg.perturbation_prompt:
                        #     input_dict = add_bbox_related_perturbations(input_dict, gt_bboxes_2d, perturbation_factor=cfg.perturbation_factor, device_id=device_id)
                    if cfg.inference_with_box_prompt:
                        input_dict['boxes_coords'] = torch.stack([data['prepare_for_dsam'][i]['boxes_coords'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                    if cfg.inference_with_depth_prompt:
                        input_dict['depth_coords'] = torch.stack([data['prepare_for_dsam'][i]['depth_coords'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                    # temp
                    input_dict['gt_masks'] = None

                    instance_masks = None
                    if 'instance_mask' in data['prepare_for_dsam'][0].keys():
                        instance_masks = torch.stack([data['prepare_for_dsam'][i]['instance_mask'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                
                    
                if cfg.merge_dino_feature:
                    input_dict["image_for_dino"] = data["image_for_dino"].to(device_id)

                only_2d_mode = len(data['prepare_for_dsam']) > 0 and torch.all(gt_bboxes_3d == -1).item()

                if cfg.provide_gt_intrinsics_inference:
                    input_dict['gt_intrinsic'] = data['K']

                # import ipdb;ipdb.set_trace()
                ret_dict = model(input_dict)
                K_for_convert = ret_dict.get('gt_intrinsic', ret_dict['pred_K']).to(device_id)
                
                if cfg.inference_with_prompt and len(data['prepare_for_dsam']) > 0:

                    pred_masks = ret_dict['masks']
                    iou_predictions = ret_dict['iou_predictions']
                    
                    # decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes(ret_dict, cfg, K_for_convert)
                    if cfg.dataset.zero_shot_dataset:
                        decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes_virtual_to_real(ret_dict, cfg, data['K'], ret_dict['pred_K'])
                    else:
                        decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes(ret_dict, cfg, K_for_convert)
                    bboxes_pred_2d_center_x = decoded_bboxes_pred_2d[..., 0]
                    bboxes_pred_2d_center_y = decoded_bboxes_pred_2d[..., 1]
                    # to check, object level or scene level
                    total_bboxes += gt_bboxes_3d.shape[0]

                    rot_mat = None
                    gt_rot_mat = None
                    if cfg.output_rotation_matrix:
                        rot_mat = rotation_6d_to_matrix(ret_dict['pred_pose_6d'])
                        # rot_mat = R_from_allocentric(K_for_convert, rot_mat, u=bboxes_pred_2d_center_x, v=bboxes_pred_2d_center_y)

                        gt_rot_mat = torch.stack([data['prepare_for_dsam'][i]['rotation_pose'].to(device_id) for i in range(len(data['prepare_for_dsam']))])

                    # if cfg.kitti_leaderboard:


                    if cfg.add_cubercnn_for_ap_inference:
                        # import ipdb;ipdb.set_trace()
                        # if len(data['prepare_for_dsam']) > 0 and 'name' in data['prepare_for_dsam'][0].keys():
                            # for i in range(len(data['prepare_for_dsam'])):
                                # print(data['prepare_for_dsam'][i]['name'])
                                # if data['prepare_for_dsam'][i]['name'] == 'truck':
                                    
                                #     decoded_bboxes_pred_3d[i][3] = 2.5
                                #     decoded_bboxes_pred_3d[i][4] = 3.5
                                #     decoded_bboxes_pred_3d[i][5] = 10
                                # if data['prepare_for_dsam'][i]['name'] == 'cyclist':
                                #     decoded_bboxes_pred_3d[i][3] = 0.6
                                #     decoded_bboxes_pred_3d[i][4] = 1.7
                                #     decoded_bboxes_pred_3d[i][5] = 1.5
                                # if data['prepare_for_dsam'][i]['name'] == 'van':
                                    
                                #     decoded_bboxes_pred_3d[i][3] = 2.0
                                #     decoded_bboxes_pred_3d[i][4] = 2.0
                                #     decoded_bboxes_pred_3d[i][5] = 5.0


                        pred_pose_for_cubercnn = calculate_pred_pose_for_cubercnn(decoded_bboxes_pred_3d, rot_mat, device_id)
                        decoded_bboxes_pred_3d_corners = compute_3d_bbox_vertices_batch(decoded_bboxes_pred_3d, rot_mat)
                        new_order = torch.tensor([5, 1, 0, 4, 6, 2, 3, 7]).to(device_id)

                        decoded_bboxes_pred_3d_corners_for_cubercnn = decoded_bboxes_pred_3d_corners[:, new_order, :]
                        

                        for i in range(len(data['prepare_for_dsam'])):
                            dict_i = {}
                            # import ipdb;ipdb.set_trace()
                            dict_i['image_id'] = data['prepare_for_dsam'][i]['image_id']
                            todo_box2d = data['prepare_for_dsam'][i]['bbox_2d'].numpy()
                            resize_ratio = max(data['original_size'].squeeze()) / max(image_h, image_w)
                            dict_i['bbox'] = (todo_box2d*resize_ratio.item()).tolist()
                            dict_i['bbox'][2] = dict_i['bbox'][2] - dict_i['bbox'][0]
                            dict_i['bbox'][3] = dict_i['bbox'][3] - dict_i['bbox'][1]
                            dict_i['category_id'] = data['prepare_for_dsam'][i]['label']
                            dict_i['score'] = data['prepare_for_dsam'][i]['score']
                            dict_i['depth'] = decoded_bboxes_pred_3d[i, 2].cpu().numpy().tolist()
                            dict_i['bbox3D'] = decoded_bboxes_pred_3d_corners_for_cubercnn[i].cpu().numpy().tolist()
                            dict_i['center_cam'] = decoded_bboxes_pred_3d[i, :3].cpu().numpy().tolist()
                            dict_i['center_2D'] = [bboxes_pred_2d_center_x[i].cpu().numpy().tolist(), bboxes_pred_2d_center_y[i].cpu().numpy().tolist()]
                            dict_i['pose'] = pred_pose_for_cubercnn[i].cpu().numpy().tolist()
                            dict_i['dimensions'] = decoded_bboxes_pred_3d[i, 3:6].cpu().numpy().tolist()
                            dict_i['area'] = dict_i['bbox'][2] * dict_i['bbox'][3] 
                            dict_i['yaw'] = decoded_bboxes_pred_3d[i, 6].cpu().numpy().tolist()
                            omni3d_result.append(dict_i)
                            # print(omni3d_result)


                    instance_ids = [dataset_name + '_' + f"{data['prepare_for_dsam'][i]['instance_id']}" for i in range(len(data['prepare_for_dsam']))]
                    if cfg.model.original_sam:
                        save_mask_images(pred_masks, iou_predictions ,image_h, image_w, gt_bboxes_2d, data['images'], instance_ids, save_root = cfg.exp_dir)
                    # visualize
                    # import ipdb;ipdb.set_trace()
                    if device_id == 0 and iter <= cfg.visualize_num:
                        # import ipdb;ipdb.set_trace()
                        # save_mask_images(pred_masks, iou_predictions ,image_h, image_w, gt_bboxes_2d, data['images'], instance_ids, save_root = cfg.exp_dir, debug_mode = True)
                        # print(decoded_bboxes_pred_3d)
                        # import ipdb;ipdb.set_trace()
                        # save_predicted_point_prompts(ret_dict['point_prompts'], data['images'], image_h, image_w)
                        input_dict['point_coords'] = torch.stack([data['prepare_for_dsam'][i]['point_coords'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                        # if cfg.model.multi_level_box_output > 1:
                        #     save_multi_level_pred_bbox(cfg, data['images'], input_dict['point_coords'][:, 0, :],
                        #         K_for_convert, data['K'], image_h, image_w, 
                        #         gt_bboxes_2d, gt_bboxes_3d, gt_center_2d, gt_rot_mat, 
                        #         dataset_name, epoch, iter, 
                        #         ret_dict['pred_bbox_2d_tensor_all'], 
                        #         ret_dict['pred_bbox_3d_depth_tensor_all'],
                        #         ret_dict['pred_bbox_3d_dims_tensor_all'],
                        #         ret_dict['pred_bbox_3d_alpha_cls_tensor_all'],
                        #         ret_dict['pred_pose_6d_tensor_all'],
                        #         ret_dict['pred_box_ious'],)
                        
                        vis_prompt_func_2(
                            cfg, data['images'], 
                            input_dict['point_coords'][:, 0, :], 
                            K_for_convert, data['K'], 
                            image_h, image_w, 
                            pred_masks, 
                            gt_bboxes_2d, 
                            decoded_bboxes_pred_2d, 
                            gt_bboxes_3d, 
                            decoded_bboxes_pred_3d, 
                            gt_center_2d,
                            ret_dict['pred_center_2d'],
                            gt_rot_mat,
                            rot_mat,
                            dataset_name,
                            epoch, iter, ret_dict.get('pred_box_ious', None))
                            

                        vis_prompt_func(
                            cfg, data['images'], 
                            input_dict['point_coords'][:, 0, :], 
                            K_for_convert, data['K'], 
                            image_h, image_w, 
                            pred_masks, 
                            gt_bboxes_2d, 
                            decoded_bboxes_pred_2d, 
                            gt_bboxes_3d, 
                            decoded_bboxes_pred_3d, 
                            gt_center_2d,
                            ret_dict['pred_center_2d'],
                            gt_rot_mat,
                            rot_mat,
                            dataset_name,
                            epoch, iter)
                        
                if cfg.inference_with_depth:

                    intrinsic_gt = data['K']
                    intrinsic_pred = ret_dict['pred_K']

                    depth_map = ret_dict['depth_maps'][:, :image_h, :image_w]
                    depth_gt = data['depth'].to(device_id)[:, :image_h, :image_w]
                    masks = data['masks'].to(device_id)[:, :image_h, :image_w]                
                    # import ipdb;ipdb.set_trace()
                    if not cfg.model.enable_dense_prompt:
                        compute_metrics(
                            scene_level_metrics, 
                            object_level_metrics, 
                            depth_gt, depth_map, 
                            masks, intrinsic_pred[0], 
                            intrinsic_gt[0], image_h, image_w,
                            decoded_bboxes_pred_2d = decoded_bboxes_pred_2d if cfg.inference_with_prompt and len(data['prepare_for_dsam']) > 0 else None,
                            gt_bboxes_2d = gt_bboxes_2d if cfg.inference_with_prompt and len(data['prepare_for_dsam']) > 0 else None,
                            decoded_bboxes_pred_3d = decoded_bboxes_pred_3d if cfg.inference_with_prompt and len(data['prepare_for_dsam']) > 0 else None,
                            gt_bboxes_3d = gt_bboxes_3d if cfg.inference_with_prompt and len(data['prepare_for_dsam']) > 0 else None,
                            rot_mat = rot_mat if cfg.inference_with_prompt and len(data['prepare_for_dsam']) > 0 else None,
                            gt_rot_mat = gt_rot_mat if cfg.inference_with_prompt and len(data['prepare_for_dsam']) > 0 else None)
                    # import ipdb;ipdb.set_trace()
                    if iter < 5 and device_id == 0:
                        depth_gt[depth_gt == torch.inf] = 0

                        if (epoch + 1) % cfg.eval_interval == 0:
                            save_depth_image(depth_gt[0].unsqueeze(0), f'{cfg.exp_dir}/{dataset_name}_depth_gt_{iter}.png', max_depth=int(depth_gt[0].max()))
                            save_color_image(data['images'][0].unsqueeze(0), image_h, image_w, f'{cfg.exp_dir}/{dataset_name}_raw_gt_{iter}.png')
                            save_point_cloud(depth_gt[0].unsqueeze(0), data['images'][0].unsqueeze(0), intrinsic_gt[0].unsqueeze(0), f"{cfg.exp_dir}/{dataset_name}_gt_{iter}.ply", image_h, image_w)

                        save_depth_image(depth_map[0].unsqueeze(0), f'{cfg.exp_dir}/{dataset_name}_depth_pred_{epoch}_{iter}.png', max_depth=int(depth_map[0].max()))
                        save_point_cloud(depth_map[0].unsqueeze(0), data['images'][0].unsqueeze(0), K_for_convert[0].unsqueeze(0), f"{cfg.exp_dir}/{dataset_name}_pred_{epoch}_{iter}.ply", image_h, image_w)
                
                if device_id == 0:
                    t.update(1) 

            if device_id == 0:
                t.set_description(desc=f"Val Epoch {epoch} for {dataset_name}")
    
    return scene_level_metrics, object_level_metrics, total_bboxes, total_scenes, omni3d_result
        

def trainval_sam(
    cfg,
    model,
    device_id,
    start_epoch,
    optimizer,
    scheduler,
    train_dataloader,
    val_dataloaders = None,
    logger = None,
    teacher_model = None,
    linear_align = None,
    channel_align = None,
    channel_reduction = None,):
    """The SAM training loop."""

    for epoch in range(start_epoch, cfg.num_epochs):
        if not cfg.inference_only:
            train_one_epoch(
                cfg,
                model,
                device_id,
                optimizer,
                scheduler,
                train_dataloader,
                epoch,
                logger,
                teacher_model,
                linear_align,
                channel_align,
                channel_reduction,
            )
            
            if cfg.rank == 0:
                save_checkpoint({ 
                    'epoch': epoch + 1, 
                    'state_dict': model.module.state_dict(), 
                    'optimizer': optimizer.state_dict(), 
                    'scheduler': scheduler.state_dict(), # 保存调度器的状态字典 
                    'scaler': cfg.scaler.state_dict(),  # 保存 AMP 状态
                    }, cfg.exp_dir, 'checkpoint_{}.pth'.format(epoch))
                    
            if (epoch + 1) % cfg.eval_interval != 0:
                continue

        with torch.no_grad():
            for dataset_name, val_dataloader in val_dataloaders:
                scene_level_metrics, object_level_metrics, total_bboxes, total_scenes, omni3d_result = validate_one_epoch(
                    cfg,
                    model,
                    device_id,
                    dataset_name,
                    val_dataloader,
                    epoch,
                    logger
                )
                
                for key, value in scene_level_metrics.items():
                    dist.all_reduce(scene_level_metrics[key])
                
                for key, value in object_level_metrics.items():
                    dist.all_reduce(object_level_metrics[key])
                
                dist.all_reduce(total_bboxes)
                dist.all_reduce(total_scenes)

                if cfg.rank == 0:    
                    avg_scene_level_metrics = {key: (value / total_scenes).cpu().item() for key, value in scene_level_metrics.items()} 
                    logger.info({f"{dataset_name}_{key}": value for key, value in avg_scene_level_metrics.items()})
                    if cfg.inference_with_prompt:
                        avg_object_level_metrics = {key: (value / total_bboxes).cpu().item() for key, value in object_level_metrics.items()} 
                        logger.info({f"{dataset_name}_{key}": value for key, value in avg_object_level_metrics.items()})

                if cfg.writer and cfg.rank == 0:
                    for key, value in avg_scene_level_metrics.items():
                        cfg.writer.add_scalar(f'Validation_{dataset_name}/{key}', value, epoch)
                    if cfg.inference_with_prompt:
                        for key, value in avg_object_level_metrics.items():
                            cfg.writer.add_scalar(f'Validation_{dataset_name}/{key}', value, epoch)
            
                if cfg.add_cubercnn_for_ap_inference:
                    dist.barrier()
                    import json
                
                    with open(f"temp_output_{cfg.rank}.json", "w") as json_file:
                        json.dump(omni3d_result, json_file, indent=4)  # indent 参数用于设置缩进，便于阅读

                    dist.barrier()   
                    if dist.get_rank() == 0:
                        final_json_results = []
                        for i in range(dist.get_world_size()):
                            
                            with open(f"temp_output_{i}.json", "r") as json_file:
                                omni_results_temp = json.load(json_file)
                            final_json_results += omni_results_temp
                            print(len(final_json_results))
                        with open(f"{cfg.exp_dir}/{dataset_name}{cfg.output_json_file}.json", "w") as json_file:
                            json.dump(final_json_results, json_file, indent=4)  # indent 参数用于设置缩进，便于阅读
                    dist.barrier()

def main():
    os.environ['NCCL_DEBUG'] = 'INFO'
    dist.init_process_group("nccl")
    # dist.init_process_group("nccl", timeout=timedelta(seconds=7200000)) # was 1800000
    os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Start running basic DDP example on rank {rank}.")
    # device_id = rank
    args = parse_args()
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    with open(args.config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg = Box(cfg)

    # 使用命令行参数覆盖配置文件参数
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)

    time_filename = datetime.now().strftime('%m%d-%H%M%S')
    cfg.exp_dir = os.path.join(cfg.exp_dir, time_filename)
    logger = get_mylogger(log_dir=cfg.exp_dir)
    cfg.rank = rank
    cfg.world_size = world_size
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.exp_dir, 'tensorboard_logs'))
    else:
        writer = None
    cfg.writer = writer

    logger.info(cfg)

    # prepare dataset
    # transform_train, transform_test = get_monu_transform()
    # train_coco = MyCOCODataset(cfg, transform = transform_train, mode = 'train')
    # val_coco = MyCOCODataset(cfg, transform = transform_test, mode = 'val')

    transform_train, transform_test = get_depth_transform()
    if cfg.tune_with_prompt:
        train_dataset = Stage2Dataset(cfg, transform = transform_train, mode = 'train')
    else:
        train_dataset = Stage1Dataset(cfg, transform = transform_train, mode = 'train')

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_sampler, collate_fn=collector)

    val_loaders = []
    for dataset_name in cfg.dataset.val.keys():
        if cfg.inference_with_prompt:
            val_dataset = Stage2Dataset(cfg, transform=transform_test, mode='val', dataset_name=dataset_name)
        else:
            val_dataset = Stage1Dataset(cfg, transform=transform_test, mode='val', dataset_name=dataset_name)

        logger.info(f"val_dataset: {val_dataset.raw_info}")

        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, sampler=val_sampler, collate_fn=collector)
        val_loaders.append((dataset_name, val_loader))
    # import ipdb;ipdb.set_trace()
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    my_sam_model = WrapModel(cfg)

    torch.cuda.empty_cache()
    cfg.scaler = GradScaler()
    if cfg.resume:
        assert os.path.isfile(cfg.resume)
        logger.info("=> loading checkpoint '{}'".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=f'cuda:{device_id}')
        start_epoch = checkpoint['epoch']
        # start_epoch = 0    
        
        new_model_dict = my_sam_model.state_dict()
        for k,v in new_model_dict.items():
            if k in checkpoint['state_dict'].keys() and checkpoint['state_dict'][k].size() == new_model_dict[k].size():
                new_model_dict[k] = checkpoint['state_dict'][k].detach()
        my_sam_model.load_state_dict(new_model_dict)

        logger.info("=> loaded checkpoint '{}' (epoch {})" .format(cfg.resume, checkpoint['epoch']))
    else: 
        start_epoch = 0    
        
        if cfg.pretrain_depth:
            assert os.path.isfile(cfg.pretrain_depth)
            logger.info("=> loading depth pretrain checkpoint '{}'".format(cfg.pretrain_depth))
            checkpoint = torch.load(cfg.pretrain_depth, map_location=f'cuda:{device_id}')
            
            new_model_dict = my_sam_model.state_dict()
            for k,v in new_model_dict.items():
                if k in checkpoint['state_dict'].keys() and checkpoint['state_dict'][k].size() == new_model_dict[k].size():
                    new_model_dict[k] = checkpoint['state_dict'][k].detach()
            my_sam_model.load_state_dict(new_model_dict)
        
        if cfg.merge_dino_feature and cfg.load_unidepth_ckpt:
            checkpoint = torch.load(cfg.unidepth_path, map_location=f'cuda:{device_id}')
            check_list = {k:0 for k in checkpoint.keys()}
            new_model_dict = my_sam_model.state_dict()
            for k,v in new_model_dict.items():
                
                if 'sam.image_encoder.dino' in k:
                    # import ipdb;ipdb.set_trace()
                    if 'pixel_encoder' + k.split('sam.image_encoder.dino')[1] in checkpoint.keys():
                        # if k in checkpoint.keys() and checkpoint[k].size() == new_model_dict[k].size():
                        new_key = 'pixel_encoder' + k.split('sam.image_encoder.dino')[1]
                        new_model_dict[k] = checkpoint[new_key].detach()
                        check_list[new_key] = 1
                elif 'sam.image_encoder.depth_head' in k:
                    if 'pixel_decoder' + k.split('sam.image_encoder.depth_head')[1] in checkpoint.keys():
                        new_key = 'pixel_decoder' + k.split('sam.image_encoder.depth_head')[1]
                        new_model_dict[k] = checkpoint[new_key].detach()
                        check_list[new_key] = 1
            my_sam_model.load_state_dict(new_model_dict)

    my_sam_model.to(device_id)
    my_sam_model.setup()
    if cfg.use_ema:
        ema = ExponentialMovingAverage(parameters=my_sam_model.parameters(), decay=0.99)
        cfg.ema = ema
    my_sam_model = DDP(my_sam_model, device_ids=[device_id], find_unused_parameters=True)

    if cfg.opt.type == 'v2':
        optimizer, scheduler = configure_opt_v2(cfg, my_sam_model, train_loader)
    elif cfg.opt.type == 'v3':
        optimizer, scheduler = configure_opt_v3(cfg, my_sam_model, train_loader)
    elif cfg.opt.type == 'v4':
        optimizer, scheduler = configure_opt_v4(cfg, my_sam_model, train_loader)
    else:
        optimizer, scheduler = configure_opt_v1(cfg, my_sam_model, train_loader)
    if cfg.resume and not cfg.inference_only:
        if cfg.resume_scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])
            # import ipdb;ipdb.set_trace()
            optimizer.load_state_dict(checkpoint['optimizer'])
            cfg.scaler.load_state_dict(checkpoint['scaler']) # 加载 AMP 状态
    # load DepthAnythingV2 large as teacher model

    if cfg.teacher_model:
        from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
        teacher_model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
         }

        encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

        teacher_model = DepthAnythingV2(**teacher_model_configs[encoder])
        teacher_model.load_state_dict(torch.load(f'/cpfs01/user/jianghaoran/detany3d/DetAny3D0827/DepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        teacher_model = teacher_model.to(device_id).eval()
        linear_align = nn.Linear(1280, 1024).to(device_id)
        channel_align = nn.Conv2d(1280, 1024, kernel_size=1).to(device_id)
        linear_align = DDP(linear_align, device_ids=[device_id], find_unused_parameters=True)
        channel_align = DDP(channel_align, device_ids=[device_id], find_unused_parameters=True)

        #version2
        channel_reduction = nn.Conv2d(256, 1, kernel_size=1).to(device_id)
        channel_reduction = DDP(channel_reduction, device_ids=[device_id], find_unused_parameters=True)

    else:
        teacher_model = None
        linear_align = None
        channel_align = None
        channel_reduction = None

    trainval_sam(cfg, my_sam_model, device_id, start_epoch, optimizer, scheduler, train_loader, val_dataloaders = val_loaders, logger = logger, teacher_model = teacher_model, linear_align = linear_align, channel_align = channel_align, channel_reduction = channel_reduction)


if __name__ == '__main__':  
    main()