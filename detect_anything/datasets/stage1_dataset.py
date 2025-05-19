from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from detect_anything.datasets import transforms_shir as transforms

import torch.nn.functional as F

import cv2
from PIL import Image
import sys
import pickle
import numpy as np
from detect_anything.utils.transforms import ResizeLongestSide
from detect_anything.utils.amg import batched_mask_to_box
from detect_anything.mylogger import *
from copy import deepcopy
from einops import rearrange

import matplotlib
import os

from detect_anything.datasets.utils import *

class Stage1Dataset(Dataset):

    def __init__(self, 
                 cfg,
                 transform,
                 mode,
                 dataset_name=None, # for val different dataset respectively
                 pixel_mean = [123.675, 116.28, 103.53],
                 pixel_std = [58.395, 57.12, 57.375],):

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

    def __getitem__(self, index):
        # for dataset_name in cfg.dataset.val.keys():
        #     if dataset_name == 'sunrgbd':
        
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

        K = pkl_now[true_index]['K'].astype(np.float32)
        K = torch.tensor(K)
        
        depth_path = pkl_now[true_index]['depth_path']
        if depth_path.endswith('.png'):
            if 'ScanNet_formal_unzip' in depth_path and 'raw.png' not in depth_path:
                depth_path = depth_path.replace('.png', 'raw.png')
            depth = np.array(Image.open(depth_path)).astype(np.float32)
        elif depth_path.endswith('.npy'):
            depth = np.load(depth_path).astype(np.float32)
        else:
            print(f'{depth_path} is not a valid depth path')
            raise NotImplementedError
        
        if self.mode == 'train':
            depth = depth / self.cfg.dataset.train[dataset_name].metric_scale
        else:
            depth = depth / self.cfg.dataset.val[dataset_name].metric_scale
        img_path = pkl_now[true_index]['img_path']
        todo_img = cv2.imread(img_path)
        todo_img = cv2.cvtColor(todo_img, cv2.COLOR_BGR2RGB) # (1208, 1920, 3)
        # import pdb;pdb.set_trace()
        original_size = tuple(todo_img.shape[:-1]) #(1208, 1920)
        todo_img, depth = self.transform(todo_img, depth) #torch.Size([3, 825, 1014])
 
        cropped_size = tuple(todo_img.shape[1:3]) #(825, 1014)
        # import ipdb;ipdb.set_trace()
        # print(K)
        K[0, 0, 2] = K[0, 0, 2] + (cropped_size[1] - original_size[1]) / 2
        K[0, 1, 2] = K[0, 1, 2] + (cropped_size[0] - original_size[0]) / 2
        # print(K)
        raw_img = todo_img
        todo_img = todo_img.unsqueeze(0) #torch.Size([1, 3, 825, 1014]) here
        todo_img = self.sam_trans.apply_image_torch(todo_img)  #torch.Size([1, 3, 833, 1024])
        depth = self.sam_trans.apply_depth_torch(depth.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        ## after resize
        # import ipdb;ipdb.set_trace()
        mask = torch.ones_like(depth)
        mask[depth == torch.inf] = 0
        mask[torch.isnan(depth)] = 0
        if self.mode == 'train':
            if self.cfg.dataset.train[dataset_name].max_distance:
                mask[depth > self.cfg.dataset.train[dataset_name].max_distance] = 0
            if self.cfg.dataset.train[dataset_name].min_distance:
                mask[depth < self.cfg.dataset.train[dataset_name].min_distance] = 0
        else:
            if self.cfg.dataset.val[dataset_name].max_distance:
                mask[depth > self.cfg.dataset.val[dataset_name].max_distance] = 0
            if self.cfg.dataset.val[dataset_name].min_distance:
                mask[depth < self.cfg.dataset.val[dataset_name].min_distance] = 0

        before_pad_size = tuple(todo_img.shape[-2:]) #(833, 1024)
        resize_ratio = before_pad_size[1] / cropped_size[1] #1.009861932938856

        # print(resize_ratio)
        K[0, 0] = K[0, 0] * resize_ratio
        K[0, 1] = K[0, 1] * resize_ratio
        if self.cfg.vit_pad_mask:
            vit_pad_size = (before_pad_size[0] // 16, before_pad_size[1] // 16)
        else:
            vit_pad_size = (64, 64)

        todo_img = self.preprocess(todo_img).squeeze(0) # torch.Size([3, 1024, 1024])
        depth_padded = torch.ones_like(todo_img[0])
        depth_mask_padded = torch.zeros_like(depth_padded)
        depth_padded[:before_pad_size[0], :before_pad_size[1]] = depth
        depth_mask_padded[:before_pad_size[0], :before_pad_size[1]] = mask

        return {
            "images": todo_img,
            # "raw_images": raw_img,
            "masks": depth_mask_padded,
            'vit_pad_size': torch.tensor(vit_pad_size),
            "K": K.squeeze(0),
            "depth": depth_padded,
            "before_pad_size": torch.Tensor(before_pad_size),
        }

    def __len__(self):
        return self.idx_cum[-1]

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
        # print(x)
        # # Pad
        h, w = x.shape[-2:]
        padh = self.cfg.model.pad - h
        padw = self.cfg.model.pad - w
        x = F.pad(x, (0, padw, 0, padh))
        return x