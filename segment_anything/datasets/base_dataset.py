import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import cv2
import pickle
import numpy as np
from PIL import Image
from copy import deepcopy

from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.datasets.utils import *
from shapely.geometry import MultiPoint
from shapely.geometry import box

import matplotlib
import os
import json
import math
from pyquaternion import Quaternion
import random

class BaseDataset(Dataset):

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

    def _get_relative_instance(self, index):

        idx = 0
        for i, i_len in enumerate(self.idx_cum):
            if index >= i_len:
                idx = i + 1
        if idx > 0:
            true_index = index - self.idx_cum[idx - 1]
        else:
            true_index = index

        pkl_now = self.pkl_list[idx]
        dataset_name = self.dataset_name_list[idx]
        instance = pkl_now[true_index]

        return pkl_now, dataset_name, instance
    
    def _load_depth(self, depth_path):

        if depth_path[-4:] == '.png':
            depth = np.array(Image.open(depth_path)).astype(np.float32)
            if self.mode == 'train':
                depth = depth / self.cfg.dataset.train[dataset_name].metric_scale
            else:
                depth = depth / self.cfg.dataset.val[dataset_name].metric_scale
        elif depth_path[-4:] == '.npy':
            depth = np.load(depth_path).astype(np.float32)
        
        return depth
    
    def process_K_transform(self, K, original_size, cropped_size = None, resized_size = None):
        pass


    def __getitem__(self, index):

        pkl_now, dataset_name, instance = self._get_relative_instance(index)

        K = torch.tensor(instance['K'].astype(np.float32))

        img_path = instance['img_path']
        todo_img = cv2.imread(img_path)
        todo_img = cv2.cvtColor(todo_img, cv2.COLOR_BGR2RGB)
        original_size = tuple(todo_img.shape[:-1])
        
        depth_path = instance['depth_path']
        depth = self._load_depth(depth_path)
        
        img, depth = self.transform(todo_img, depth)

        cropped_size = tuple(img.shape[1:3])
        cropped_blank_H = int((original_size[0] - cropped_size[0]) / 2)
        cropped_blank_W = int((original_size[1] - cropped_size[1]) / 2)

        # bx, by will change if cropped
        K[0, 0, 2] = K[0, 0, 2] - cropped_blank_W
        K[0, 1, 2] = K[0, 1, 2] - cropped_blank_H

        # resize the long edge to target size
        img = self.sam_trans.apply_image_torch(img.unsqueeze(0))
        
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

            # for object detection
            # cropped_blank_H = int((original_size[0] - before_pad_size[0]) / 2)
            # cropped_blank_W = int((original_size[1] - before_pad_size[1]) / 2)
        
        raw_image = img.clone().squeeze(0)
        # nomalize and pad for sam
        img_for_sam = self.preprocess(img).squeeze(0)

        # generate data for object detection
        prepare_for_dsam = self.generate_obj_list(instance, cropped_blank_H, cropped_blank_W, K, before_pad_size, original_size)
        
        # random choose another frame
        if len(prepare_for_dsam) == 0:
            return self.__getitem__(random.randint(0, self.idx_cum[-1]-1))
        
        # calculate the vit pad size for depth head
        if self.cfg.model.vit_pad_mask:
            vit_pad_size = (before_pad_size[0] // self.cfg.model.image_encoder.patch_size, before_pad_size[1] // self.cfg.model.image_encoder.patch_size)
        else:
            vit_pad_size = (self.cfg.model.pad // self.cfg.model.image_encoder.patch_size, self.cfg.model.pad // self.cfg.model.image_encoder.patch_size)
        
        # padding others
        depth_padded = torch.zeros_like(img_for_sam[0])
        depth_mask_padded = torch.zeros_like(depth_padded)

        depth_padded[:before_pad_size[0], :before_pad_size[1]] = depth
        depth_mask_padded[:before_pad_size[0], :before_pad_size[1]] = 1
        depth_mask_padded[depth_padded == torch.inf] = 0
        depth_mask_padded[depth_padded == torch.nan] = 0

        if self.mode == 'train':
            if self.cfg.dataset.train[dataset_name].max_distance:
                depth_mask_padded[depth_padded > self.cfg.dataset.train[dataset_name].max_distance] = 0
            if self.cfg.dataset.train[dataset_name].min_distance:
                depth_mask_padded[depth_padded < self.cfg.dataset.train[dataset_name].min_distance] = 0
        else:
            if self.cfg.dataset.val[dataset_name].max_distance:
                depth_mask_padded[depth_padded > self.cfg.dataset.val[dataset_name].max_distance] = 0
            if self.cfg.dataset.val[dataset_name].min_distance:
                depth_mask_padded[depth_padded < self.cfg.dataset.val[dataset_name].min_distance] = 0
                
        depth_padded[depth_padded == torch.inf] = 0
        depth_padded[depth_padded == torch.nan] = 0

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
    
    def crop_hw(self, img, depth, K):
        
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

        # 更新相机内参 K
        K_cropped = K.clone()  # 假设 K 是一个 numpy 数组
        K_cropped[0, 0, 2] -= (start_w)  # 更新 x 坐标
        K_cropped[0, 1, 2] -= (start_h)  # 更新 y 坐标

        return img_cropped.unsqueeze(0), depth_cropped, K_cropped

