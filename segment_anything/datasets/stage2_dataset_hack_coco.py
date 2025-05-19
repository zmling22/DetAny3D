from torch.utils.data import Dataset, DataLoader
import torch

import torch.nn.functional as F
import cv2
import pickle
import numpy as np
from PIL import Image

from segment_anything.utils.transforms import ResizeLongestSide
from copy import deepcopy
from segment_anything.datasets.utils import *
from shapely.geometry import MultiPoint
from shapely.geometry import box

import matplotlib
import os
import json
import math
from pyquaternion import Quaternion
import random
import json
import h5py


class Stage2Dataset(Dataset):

    def __init__(self, 
                 cfg,
                 transform,
                 mode,
                 # for val different dataset respectively
                 dataset_name=None,
                 ):

        self.dataset_name_list = []
        self.pkl_path_list = []
        self.len_idx = []
        self.pkl_list = []

        self.sam_trans = ResizeLongestSide(cfg.model.pad)
        self.transform = transform

        if mode == 'train':
            dataset_dict = cfg.dataset.train
        elif mode == 'val':
            dataset_dict = cfg.dataset.val
        else:
            raise NotImplementedError('no test mode yet')
        
        if dataset_name is not None:
            # 仅加载指定数据集 (val)
            dataset_info = dataset_dict[dataset_name]
            self._load_single_dataset(dataset_name, dataset_info)
        else:
            # 加载所有数据集 (train)
            for dataset_name in dataset_dict.keys():
                dataset_info = dataset_dict[dataset_name]
                self._load_single_dataset(dataset_name, dataset_info)
        self.raw_info = [(dataset_name, num_samples) for dataset_name, num_samples in zip(self.dataset_name_list, self.len_idx)]

        self.idx_cum = np.cumsum(self.len_idx)
        self.pixel_mean = torch.Tensor(cfg.dataset.pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.dataset.pixel_std).view(-1, 1, 1)
        self.cfg = cfg
        self.mode = mode

    def _load_single_dataset(self, dataset_name, dataset_info):
        """加载单个数据集的功能，供重复使用"""
        self.dataset_name_list.append(dataset_name)
        self.pkl_path_list.append(dataset_info.pkl_path)
        
        with open(dataset_info.pkl_path, 'rb') as f:
            tmp_pkl = pickle.load(f)[dataset_info.range.begin:dataset_info.range.end:dataset_info.range.interval]
            self.pkl_list.append(tmp_pkl)
        self.len_idx.append(len(tmp_pkl))
        print(f"Dataset: {dataset_name}, Number of samples: {len(tmp_pkl)}")
        # temp add coco here, todo: change to general
        if dataset_name == 'coco':
            from segment_anything.datasets.coco_utils import COCO
            self.annFile = '{}/annotations/instances_{}.json'.format(dataset_info.dataDir, dataset_info.dataType)
            self.coco = COCO(self.annFile)
            self.data_list = self.coco.getImgIds()

    def _get_relative_index(self, index):

        idx = 0
        for i, i_len in enumerate(self.idx_cum):
            if index >= i_len:
                idx = i + 1
        if idx > 0:
            true_index = index - self.idx_cum[idx - 1]
        else:
            true_index = index

        return idx, true_index
    
    def _load_depth(self, depth_path, dataset_name, img):

        if depth_path is None:
            height, width = img.shape[:2]
            depth = np.zeros((height, width), dtype=np.float32)
        elif depth_path[-4:] == '.png':
            depth = np.array(Image.open(depth_path)).astype(np.float32)
            if self.mode == 'train':
                depth = depth / self.cfg.dataset.train[dataset_name].metric_scale
            else:
                depth = depth / self.cfg.dataset.val[dataset_name].metric_scale
        elif depth_path[-4:] == '.npy':
            depth = np.load(depth_path).astype(np.float32)
        elif depth_path[-4:] == 'hdf5':
            assert 'hypersim' in dataset_name, 'only hypersim support now'
            intWidth = 1024
            intHeight = 768
            fltFocal = 886.81

            hf = h5py.File(depth_path, 'r')
            n1 = hf.get('dataset')[:]
            npyDistance = np.array(n1)

            npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
            npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
            npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
            npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

            depth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
            depth = depth.astype(np.float32)
        else:
            raise NotImplementedError

        depth[np.isnan(depth)] = 0
        depth[np.isinf(depth)] = 0
        return depth

    def __getitem__(self, index):

        idx, true_index = self._get_relative_index(index)
       
        pkl_now = self.pkl_list[idx]
        dataset_name = self.dataset_name_list[idx]
        if dataset_name == 'coco':
            return self.get_coco_item(true_index, pkl_now, dataset_name)
        instance = pkl_now[true_index]

        K = instance['K'].astype(np.float32)
        K = torch.tensor(K)
        
        img_path = instance['img_path']
        if self.cfg.hack_img_path:
            img_path = self.cfg.hack_img_path
        todo_img = cv2.imread(img_path)
        todo_img = cv2.cvtColor(todo_img, cv2.COLOR_BGR2RGB)
        original_size = tuple(todo_img.shape[:-1])
        
        depth_path = instance['depth_path']
        if self.cfg.hack_img_path:
            depth_path = None
        depth = self._load_depth(depth_path, dataset_name, todo_img)
        
        img, depth = self.transform(todo_img, depth)

        cropped_size = tuple(img.shape[1:3])
        cropped_blank_H = int((original_size[0] - cropped_size[0]) / 2)
        cropped_blank_W = int((original_size[1] - cropped_size[1]) / 2)

        # bx, by will change if cropped
        K[0, 0, 2] = K[0, 0, 2] - cropped_blank_W
        K[0, 1, 2] = K[0, 1, 2] - cropped_blank_H

        # resize the long edge to target size
        img = img.unsqueeze(0)
        img = self.sam_trans.apply_image_torch(img)
        
        before_pad_size = tuple(img.shape[-2:])
        resize_ratio = before_pad_size[1] / cropped_size[1]

        # fx, fy, bx, by will change if resized
        K[0, 0] = K[0, 0] * resize_ratio
        K[0, 1] = K[0, 1] * resize_ratio

        depth = self.sam_trans.apply_depth_torch(depth.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        if self.cfg.merge_dino_feature:
            # insure the short edge is divisible by 112
            img, depth, K = self.crop_hw(img, depth, K)
            before_pad_size = tuple(img.shape[2:])
        
        raw_image = img.clone().squeeze(0)
        # nomalize and pad for sam
        img_for_sam = self.preprocess(img).squeeze(0)

        # generate data for object detection
        prepare_for_dsam = self.generate_obj_list(instance, cropped_blank_H, cropped_blank_W, K, before_pad_size, original_size, raw_image, dataset_name)
        
        # random choose another frame
        if len(prepare_for_dsam) == 0:
            print(img_path)
            print('Warning: no valid object detected, return another sample')
            return self.__getitem__(random.randint(0, self.idx_cum[-1]-1))
        
        if len(prepare_for_dsam) > 50:
            prepare_for_dsam = prepare_for_dsam[:50]
        
        # calculate the vit pad size for depth head
        if self.cfg.model.vit_pad_mask:
            vit_pad_size = (before_pad_size[0] // self.cfg.model.image_encoder.patch_size, before_pad_size[1] // self.cfg.model.image_encoder.patch_size)
        else:
            vit_pad_size = (self.cfg.model.pad // self.cfg.model.image_encoder.patch_size, self.cfg.model.pad // self.cfg.model.image_encoder.patch_size)
        
        # padding depth
        depth_padded, depth_mask_padded = self.process_depth(img_for_sam, depth, before_pad_size, dataset_name)
        
        return_dict = {
            "images": img_for_sam,
            "masks": depth_mask_padded,
            'vit_pad_size': torch.tensor(vit_pad_size),
            "K": K.squeeze(0),
            "depth": depth_padded,
            "before_pad_size": torch.Tensor(before_pad_size),

            # stage2 related params here
            "prepare_for_dsam": prepare_for_dsam,

        }

        if self.cfg.merge_dino_feature:

            # post process image for dino, without padding
            img_for_dino = self.preprocess_dino(img).squeeze(0)

            return_dict.update({
                # input for dino
                "image_for_dino": img_for_dino,})
        
        return return_dict

    def __len__(self):
        return self.idx_cum[-1]
    
    def get_coco_item(self, index, pkl, dataset_name = 'coco'):
        assert dataset_name == 'coco', 'dataset_name must be coco'
        anns = pkl[index]['anns_all_img']
        mask_all_image = self.coco.annlistToMask(anns)

        img_path = pkl[index]['img_path']
        img_path = '/cpfs01/user/zhanghanxue/segment-anything/000000486438.jpg'
        todo_img = cv2.imread(img_path)
        todo_img = cv2.cvtColor(todo_img, cv2.COLOR_BGR2RGB)
        original_size = tuple(todo_img.shape[:-1])

        K = np.array([[[2 * original_size[0], 0, original_size[1] / 2],
                        [0, 2 * original_size[0], original_size[0] / 2],
                        [0, 0, 1]]]).astype(np.float32)
        K = torch.tensor(K)

        depth = self._load_depth(None, dataset_name, todo_img)

        img, depth = self.transform(todo_img, depth)

        cropped_size = tuple(img.shape[1:3])
        cropped_blank_H = int((original_size[0] - cropped_size[0]) / 2)
        cropped_blank_W = int((original_size[1] - cropped_size[1]) / 2)

        K[0, 0, 2] = K[0, 0, 2] - cropped_blank_W
        K[0, 1, 2] = K[0, 1, 2] - cropped_blank_H
        

        img = img.unsqueeze(0)
        img = self.sam_trans.apply_image_torch(img)

        before_pad_size = tuple(img.shape[-2:])
        resize_ratio = before_pad_size[1] / cropped_size[1]

        K[0, 0] = K[0, 0] * resize_ratio
        K[0, 1] = K[0, 1] * resize_ratio

        depth = self.sam_trans.apply_depth_torch(depth.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        if self.cfg.merge_dino_feature:
            # insure the short edge is divisible by 112
            img, depth, K = self.crop_hw(img, depth, K)
            before_pad_size = tuple(img.shape[2:])

        raw_image = img.clone().squeeze(0)
        img_for_sam = self.preprocess(img).squeeze(0)
        prepare_for_dsam = []
        if random.random() < 0.8:
            two_point_prompt = False
        else:
            two_point_prompt = True
        hack_boxes = np.array([[ 27.2420, 172.4036, 627.3259, 424.3036],
        [120.7209, 306.2226, 342.0404, 425.3015],
        [344.0049, 313.1316, 573.3167, 425.2663],
        [ 27.0154, 249.7441, 229.3946, 376.6555],
        [354.5780, 187.9767, 559.3692, 296.9029],
        [206.3441, 171.1475, 381.6691, 275.2636],
        [443.7803, 264.3295, 626.9164, 372.1801],
        [256.8008, 277.9950, 437.1614, 370.9505],
        [156.4904, 226.2845, 272.7968, 307.6314],
        [499.6505, 120.1068, 601.3515, 182.8123]])
        
        for idx_instance, ann in enumerate(anns):
            # if ann['iscrowd'] == 1:
            #     continue
            
            mask = self.coco.annToMask(ann)
            seg_mask = torch.tensor(mask).to(torch.float32)
            seg_mask = self.sam_trans.apply_mask_torch(seg_mask.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            mask_positions = np.argwhere(mask == 1)
            # if len(mask_positions) == 0:
            #     continue
            selected_positions = mask_positions[np.random.choice(mask_positions.shape[0], self.cfg.num_point_prompts if self.mode == 'train' else 1, replace=True)]
            selected_positions = [[pos[1], pos[0]] for pos in selected_positions]
            point_coords_tensor = torch.tensor(selected_positions, dtype=torch.int)
            point_coords_tensor = self.sam_trans.apply_coords_torch(point_coords_tensor, original_size).to(torch.int)

            if two_point_prompt and self.mode == 'train':
                if point_coords_tensor.shape[0] > 1:
                    tmp_point_coords_list = [point_coords_tensor[i:i+2, ...] for i in range(point_coords_tensor.shape[0] // 2)]
                else:
                    tmp_point_coords_list = [point_coords_tensor.repeat(2, 1)]
            else:
                tmp_point_coords_list = [point_coords_tensor[i:i+1, ...] for i in range(point_coords_tensor.shape[0])] 
            bbox_2d_tensor = torch.tensor(ann['bbox'])
            bbox_2d_tensor[2] = bbox_2d_tensor[0] + bbox_2d_tensor[2]
            bbox_2d_tensor[3] = bbox_2d_tensor[1] + bbox_2d_tensor[3]
            bbox_2d_tensor = torch.tensor(hack_boxes[idx_instance])

            bbox_2d_tensor = self.sam_trans.apply_boxes_torch(bbox_2d_tensor, original_size).to(torch.int).squeeze(0)

            # if bbox_2d_tensor[2] - bbox_2d_tensor[0] < 5 or bbox_2d_tensor[3] - bbox_2d_tensor[1] < 5:
            #     # print('a potential risk of bbox size')
            #     continue
            # if bbox_2d_tensor[3] - bbox_2d_tensor[1] < 0.0625 * before_pad_size[0]:
            #     # print('a potential risk of bbox size')
            #     continue
            # import ipdb;ipdb.set_trace()
            for coord in tmp_point_coords_list:
                todo_dict = {
                    "bbox_2d": bbox_2d_tensor,
                    "point_coords": coord,
                    "bbox_3d": torch.tensor([-1, -1, -1, -1, -1, -1, -1]),
                    "center_2d": torch.tensor([-1, -1]),
                    "instance_id": ann['id'],
                    "instance_mask": seg_mask,
                    }
                if self.cfg.output_rotation_matrix:
                    todo_dict['rotation_pose'] = torch.eye(3)
                
                prepare_for_dsam.append(
                    todo_dict
                )
            if idx_instance == len(hack_boxes) - 1:
                break
                # im andpoint_prompt.jpg', to_draw)

        if len(prepare_for_dsam) == 0:
            print(img_path)
            print('Warning: no valid object detected, return another sample')
            return self.__getitem__(random.randint(0, self.idx_cum[-1]-1))
        
        if len(prepare_for_dsam) > 50:
            prepare_for_dsam = prepare_for_dsam[:50]
        
        if self.cfg.model.vit_pad_mask:
            vit_pad_size = (before_pad_size[0] // self.cfg.model.image_encoder.patch_size, before_pad_size[1] // self.cfg.model.image_encoder.patch_size)
        else:
            vit_pad_size = (self.cfg.model.pad // self.cfg.model.image_encoder.patch_size, self.cfg.model.pad // self.cfg.model.image_encoder.patch_size)
        
        # padding depth
        depth_padded, depth_mask_padded = self.process_depth(img_for_sam, depth, before_pad_size, dataset_name)

        return_dict = {
            "images": img_for_sam,
            "masks": depth_mask_padded,
            'vit_pad_size': torch.tensor(vit_pad_size),
            "K": K.squeeze(0) if K is not None else None,
            "depth": depth_padded,
            "before_pad_size": torch.Tensor(before_pad_size),

            # stage2 related params here
            "prepare_for_dsam": prepare_for_dsam,

        }

        if self.cfg.merge_dino_feature:

            # post process image for dino, without padding
            img_for_dino = self.preprocess_dino(img).squeeze(0)

            return_dict.update({
                # input for dino
                "image_for_dino": img_for_dino,})
        
        return return_dict

    
    def generate_obj_list(self, instance, cropped_blank_H, cropped_blank_W, K, before_pad_size, original_size, raw_image, dataset_name):

        # use mask to mimick human inputs
        # import ipdb;ipdb.set_trace()
        if 'instance_masks' in instance.keys():
            masks = np.array(Image.open(instance['instance_masks']))
            if cropped_blank_H != 0 and cropped_blank_W !=0:
                masks = np.ascontiguousarray(masks[cropped_blank_H:-cropped_blank_H-1, cropped_blank_W: -cropped_blank_W-1])
            elif cropped_blank_H == 0 and cropped_blank_W != 0:
                masks = np.ascontiguousarray(masks[:, cropped_blank_W: -cropped_blank_W-1])
            elif cropped_blank_W == 0 and cropped_blank_H != 0:
                masks = np.ascontiguousarray(masks[cropped_blank_H:-cropped_blank_H-1, :])

        prepare_for_dsam = []

        if random.random() < 0.8:
            two_point_prompt = False
        else:
            two_point_prompt = True

        obj_list = instance['obj_list']
        for obj in obj_list:

            # 2d bbox should be the projection of 3d bbox to image plane
            x, y, z, w, h, l, yaw = obj['3d_bbox']
            pose = None
            if self.cfg.output_rotation_matrix:
                pose = obj['rotation_pose']
            vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, pose)
            vertices_2d = project_to_image(vertices_3d, K.squeeze(0))
            polygon_from_2d_box = MultiPoint(vertices_2d).convex_hull  

            img_canvas = box(0, 0, before_pad_size[1], before_pad_size[0]) 
            if polygon_from_2d_box.intersects(img_canvas):  
                img_intersection = polygon_from_2d_box.intersection(img_canvas)
                intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])
                min_x = min(intersection_coords[:, 0])
                min_y = min(intersection_coords[:, 1])
                max_x = max(intersection_coords[:, 0])
                max_y = max(intersection_coords[:, 1])
            else:
                # print('a potential risk of out bbox')
                continue
            # import ipdb;ipdb.set_trace()
            bbox_2d_polygon = [min_x, min_y, max_x, max_y]
            if '2d_bbox' in obj.keys():
                bbox_2d = obj['2d_bbox']
            elif '2d_bbox_tight' in obj.keys():
                if obj['2d_bbox_tight'] != [-1, -1, -1, -1]:
                    bbox_2d = obj['2d_bbox_tight']
                else:
                    bbox_2d = obj['2d_bbox_proj']
            else:
                raise ValueError('no 2d bbox found')
            
            bbox_2d_tensor = torch.tensor(bbox_2d, dtype=torch.int)
            if self.cfg.add_cubercnn_for_ap_inference and self.cfg.use_pred_cubercnn:
                bbox_2d_tensor[2] = bbox_2d_tensor[0] + bbox_2d_tensor[2]
                bbox_2d_tensor[3] = bbox_2d_tensor[1] + bbox_2d_tensor[3]
            bbox_2d_tensor = self.sam_trans.apply_boxes_torch(bbox_2d_tensor, original_size).to(torch.int).squeeze(0)
            # if '2d_bbox_tight' not in obj.keys() or obj['2d_bbox_tight'] == [-1, -1, -1, -1]:
            #     bbox_2d_tensor[0::2] = torch.clamp(bbox_2d_tensor[0::2], min=max(min_x, 0), max=min(max_x, before_pad_size[1]))
            #     bbox_2d_tensor[1::2] = torch.clamp(bbox_2d_tensor[1::2], min=max(min_y, 0), max=min(max_y, before_pad_size[0]))

            if bbox_2d_tensor[2] - bbox_2d_tensor[0] < 5 or bbox_2d_tensor[3] - bbox_2d_tensor[1] < 5:
                # print('a potential risk of bbox size')
                continue
            if bbox_2d_tensor[3] - bbox_2d_tensor[1] < 0.0625 * before_pad_size[0]:
                # print('a potential risk of bbox size')
                continue

            # calculate center 2d from 3d center
            center_2d = project_to_image(np.array([[x, y, z]]), K.squeeze(0)).squeeze(0)
            if not self.cfg.contain_edge_obj:
                if center_2d[0] < 0 or center_2d[0] > before_pad_size[1] or center_2d[1] < 0 or center_2d[1] > before_pad_size[0]:
                    continue
            
            center_2d_tensor = torch.tensor(center_2d)

            # modify yaw to [-pi, pi]
            if yaw > np.pi:
                yaw = yaw - 2 * np.pi
            if yaw < -np.pi:
                yaw = yaw + 2 * np.pi
            bbox_3d = [x, y, z, w, h, l, yaw]
            if bbox_3d[2] < 0:
                # print('a potential risk of depth error')
                continue
            bbox_3d_tensor = torch.tensor(bbox_3d)

            # mimick human prompt according to segmentation masks if mask exists
            if 'instance_masks' in instance.keys():
                target_value = obj['instance_id']
                coordinates = np.argwhere(masks == target_value)
                new_coordinates = []
                for coord in coordinates:
                    new_coordinates.append(coord)
                new_coordinates = np.array(new_coordinates)
                if len(new_coordinates) > 0:
                    # random choose a point in masks
                    random_index = np.random.choice(len(new_coordinates))
                    random_coordinate = new_coordinates[random_index][::-1]
                    human_prompt_coord = random_coordinate.copy()
                    point_coords_tensor = torch.tensor(human_prompt_coord, dtype=torch.int)
                    point_coords_tensor = self.sam_trans.apply_coords_torch(point_coords_tensor, masks.shape).to(torch.int).unsqueeze(0)
                    # customize the input prompt
                    # reverse_coor = np.array([534, 638])
                else:
                    continue
            
            else:
                human_prompt_coord = np.array([int((bbox_2d_tensor[0] + bbox_2d_tensor[2]) / 2), int((bbox_2d_tensor[1] + bbox_2d_tensor[3]) / 2)]) #* 0.5
                point_coords_tensor = torch.tensor(human_prompt_coord, dtype=torch.int).unsqueeze(0)
            # customize the input prompt
            if self.cfg.hack_point_prompt:
                point_coords_tensor = torch.tensor(self.cfg.hack_point_prompt).unsqueeze(0)
            if self.cfg.hack_box_prompt:
                bbox_2d_tensor = torch.tensor(self.cfg.hack_box_prompt)
            
            if self.cfg.perturbation_prompt and self.mode == 'train':
                # import ipdb;ipdb.set_trace()
                point_coords_tensor = add_bbox_related_perturbations(point_coords_tensor, bbox_2d_tensor, perturbation_factor=self.cfg.perturbation_factor, num_pertuerbated_points = self.cfg.num_point_prompts)
            
            instance_id = obj['instance_id']
            mask_path = f'exps/{dataset_name}_masks/{instance_id}_mask.jpg'

            if os.path.exists(mask_path) and self.cfg.generate_point_prompts_via_mask:
                obj_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                point_coords_tensor = self.get_point_coords_from_mask(mask_path, num_point_prompts = self.cfg.num_point_prompts, original_point_coord_tensor = point_coords_tensor)
            
            if two_point_prompt and self.mode == 'train':
                if point_coords_tensor.shape[0] > 1:
                    tmp_point_coords_list = [point_coords_tensor[i:i+2, ...] for i in range(point_coords_tensor.shape[0] // 2)]
                else:
                    tmp_point_coords_list = [point_coords_tensor.repeat(2, 1)]
            else:
                tmp_point_coords_list = [point_coords_tensor[i:i+1, ...] for i in range(point_coords_tensor.shape[0])]        
            # tmp_point_coords_list = [[278, 37], [296, 339], [592, 223], [142, 212], [556, 210], [869, 229], [869, 252], [337, 126], [604, 193], [137, 139], [341, 120], [149, 345], [619, 197], [384, 294], [605, 203], [303, 124], [416, 106], [403, 120], [624, 208], [160, 383], [762, 170], [138, 291], [63, 404], [527, 359], [402, 273], [343, 462], [657, 278], [472, 490], [38, 396], [804, 172], [290, 334], [793, 162], [808, 169], [533, 237], [773, 328], [12, 309], [543, 287], [431, 272], [9, 276], [884, 320], [795, 179], [449, 276]]
            # tmp_point_coords_list = [torch.tensor(coord, dtype=torch.int).unsqueeze(0) for coord in tmp_point_coords_list]
            for coord in tmp_point_coords_list:
    
                todo_dict = {
                    "bbox_2d": bbox_2d_tensor,
                    "point_coords": coord, #.unsqueeze(0),
                    "bbox_3d": bbox_3d_tensor,
                    "center_2d": center_2d_tensor,
                    "instance_id": obj['instance_id'],
                    # 'obj_mask': torch.tensor(obj_mask)
                    }

                if self.cfg.output_rotation_matrix:
                    todo_dict['rotation_pose'] = torch.tensor(obj['rotation_pose']).to(torch.float32)
                if self.cfg.add_cubercnn_for_ap_inference:
                    todo_dict['label'] = obj['label']
                    todo_dict['score'] = obj['score']
                    todo_dict['image_id'] = obj['image_id']
                    
                prepare_for_dsam.append(
                    todo_dict
                )

            # if len(prepare_for_dsam) >= 10:
            #     break

            # visualization code
            # import ipdb; ipdb.set_trace()
            # [bbox_x1, bbox_y1, bbox_x2, bbox_y2] = bbox_2d_tensor
            # coor = [(int(bbox_x1), int(bbox_y1)), (int(bbox_x2), int(bbox_y2))]
            # to_draw = raw_image.permute(1, 2, 0).type(torch.uint8).numpy()
            # to_draw = cv2.cvtColor(to_draw, cv2.COLOR_RGB2BGR)
            # cv2.circle(to_draw, (int(point_coords_tensor[0][0]),int(point_coords_tensor[0][1])), 2, (0, 0, 255), 4)
            # cv2.circle(to_draw, (int(center_2d_tensor[0]),int(center_2d_tensor[1])), 2, (255, 255, 0), 4)
            # cv2.rectangle(to_draw, coor[0], coor[1], (0, 0, 255), 2)
            # cv2.imwrite('img_with_point_prompt.jpg', to_draw)

            # x, y, z, w, h, l, yaw = bbox_3d_tensor
            # vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, obj['rotation_pose'])
            # vertices_2d = project_to_image(vertices_3d, K.squeeze(0))
            # fore_plane_center_2d = project_to_image(fore_plane_center_3d, K.squeeze(0))
            # to_draw = raw_image.permute(1, 2, 0).type(torch.uint8).numpy()
            # to_draw = cv2.cvtColor(to_draw, cv2.COLOR_RGB2BGR)
            # draw_bbox_2d(to_draw, vertices_2d)
            # cv2.circle(to_draw, fore_plane_center_2d[0].astype(int), 2, (0, 0, 255) , 1)
            # cv2.imwrite('3D_test_change_K.png', to_draw)

        return prepare_for_dsam

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.cfg.model.pad - h
        padw = self.cfg.model.pad - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def preprocess_dino(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = x / 255
        IMAGENET_DATASET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        IMAGENET_DATASET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        x = (x - IMAGENET_DATASET_MEAN) / IMAGENET_DATASET_STD

        return x
    
    def crop_hw(self, img, depth, K = None):
        
        if img.dim() == 4:
            img = img.squeeze(0)
        h, w = img.shape[1:3]  # 假设形状为 [C, H, W]
        assert max(h, w) % 112 == 0, "target_size must be divisible by 112"

        # 计算裁剪后尺寸，确保可以被 14 整除
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14

        # 计算裁剪区域的中心
        center_h, center_w = h // 2, w // 2

        # 计算裁剪的起始和结束索引
        start_h = center_h - new_h // 2
        start_w = center_w - new_w // 2

        # 按照中心裁剪图像和深度图
        img_cropped = img[:, start_h:start_h + new_h, start_w:start_w + new_w]
        depth_cropped = depth[start_h:start_h + new_h, start_w:start_w + new_w]

        K_cropped = None
        # 更新相机内参 K
        if K is not None:
            K_cropped = K.clone()  # 假设 K 是一个 numpy 数组
            K_cropped[0, 0, 2] -= (start_w)  # 更新 x 坐标
            K_cropped[0, 1, 2] -= (start_h)  # 更新 y 坐标

        return img_cropped.unsqueeze(0), depth_cropped, K_cropped

    def get_point_coords_from_mask(self, mask_path, num_point_prompts, original_point_coord_tensor, min_area=50, edge_margin=5):
        obj_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)
        
        valid_regions = []
        largest_area = 0
        largest_region_label = -1
        
        # Loop through the regions to find the largest one
        for i in range(1, num_labels):  # skip background (label=0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                valid_regions.append(i)
                if area > largest_area:
                    largest_area = area
                    largest_region_label = i

        if largest_region_label == -1:
            print("No valid regions found.", mask_path)
            return original_point_coord_tensor
        
        # Get all the points in the largest region
        points = np.column_stack(np.where(labels == largest_region_label))
        
        # Filter points to avoid edge areas
        safe_points = []
        for point in points:
            # Avoid points near the edges by checking if the point is within the "safe" region
            x, y = point
            if (x > edge_margin and x < obj_mask.shape[0] - edge_margin and
                y > edge_margin and y < obj_mask.shape[1] - edge_margin):
                safe_points.append(point)

        if not safe_points:
            print("No valid points found away from the edges.", mask_path)
            return original_point_coord_tensor

        mask_points_list = []
        if self.mode != 'train':
            num_point_prompts = 1

        for i in range(num_point_prompts):
            # Randomly choose a point from the safe points within the largest region
            selected_point = safe_points[np.random.choice(len(safe_points))][::-1]
            mask_points_list.append(selected_point)
        mask_points_list = np.array(mask_points_list)

        return torch.tensor(mask_points_list)
    
    def process_depth(self, img_for_sam, depth, before_pad_size, dataset_name):
        """
        Process the depth image with padding and apply depth mask based on dataset settings.

        Args:
            img_for_sam (Tensor): The image tensor for SAM (used for matching size).
            depth (Tensor): The depth image to be padded and processed.
            before_pad_size (tuple): The size of the depth image before padding.
            dataset_name (str): The name of the dataset to retrieve specific configuration.

        Returns:
            depth_padded (Tensor): The padded depth image.
            depth_mask_padded (Tensor): The padded depth mask indicating valid depths.
        """
        # Initialize padded depth and depth mask
        depth_padded = torch.zeros_like(img_for_sam[0])
        depth_mask_padded = torch.zeros_like(depth_padded)

        # Pad depth and mask
        depth_padded[:before_pad_size[0], :before_pad_size[1]] = depth
        depth_mask_padded[:before_pad_size[0], :before_pad_size[1]] = 1

        # Apply max/min distance filtering based on mode
        if self.mode == 'train':
            if self.cfg.dataset.train[dataset_name].max_distance:
                depth_mask_padded[depth_padded >= self.cfg.dataset.train[dataset_name].max_distance] = 0
            if self.cfg.dataset.train[dataset_name].min_distance:
                depth_mask_padded[depth_padded <= self.cfg.dataset.train[dataset_name].min_distance] = 0
        else:
            if self.cfg.dataset.val[dataset_name].max_distance:
                depth_mask_padded[depth_padded >= self.cfg.dataset.val[dataset_name].max_distance] = 0
            if self.cfg.dataset.val[dataset_name].min_distance:
                depth_mask_padded[depth_padded <= self.cfg.dataset.val[dataset_name].min_distance] = 0

        return depth_padded, depth_mask_padded


# from box import Box
# import yaml

# with open('segment_anything/configs/test_dataset_config.yaml', 'r', encoding='utf-8') as f:
#     cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
# cfg = Box(cfg)

# transforms_train, transforms_test = get_depth_transform()
# test_data = Stage2Dataset(cfg, transforms_train, 'val')
# for i in range(len(test_data)):
#     instance = test_data[i]