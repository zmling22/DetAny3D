import pickle
import cv2
import numpy as np
import torch
from einops import rearrange
import matplotlib
import os
import json
from PIL import Image
from tqdm import tqdm
stereo_pkl_dict = []
for _, dirs, _ in os.walk('/cpfs01/shared/opendrivelab/opendrivelab_hdd/haoran/drivingstereo_new/OpenDataLab___DrivingStereo/raw/train-depth-map'):
    print(dirs)
    print('here', len(dirs))
    for dir in tqdm(dirs):
        for _, _, files in os.walk('/cpfs01/shared/opendrivelab/opendrivelab_hdd/haoran/drivingstereo_new/OpenDataLab___DrivingStereo/raw/train-depth-map/'+ dir):
            print('here', len(files))
            for file in files:
                # import ipdb;ipdb.set_trace()
                tmp_dict = {}
                instance_id = file.split('.png')[0]
                
                depth_path = f'/cpfs01/shared/opendrivelab/opendrivelab_hdd/haoran/drivingstereo_new/OpenDataLab___DrivingStereo/raw/train-depth-map/'+ dir + '/' + instance_id + '.png'
                img_path = f'/cpfs01/shared/opendrivelab/opendrivelab_hdd/haoran/drivingstereo_new/OpenDataLab___DrivingStereo/raw/train-left-image/'+ dir + '/' + instance_id + '.jpg'
                # gt_calib_path = f'/cpfs01/shared/opendrivelab/opendrivelab_hdd/haoran/drivingstereo_new/OpenDataLab___DrivingStereo/raw/calib/half-image-calib/'+ dir + '.txt'
                K = np.array([[[1003.556, 0, 440.5], [0, 1006.938, 200], [0, 0, 1]]]).reshape(1, 3, 3)

                tmp_dict['depth_path'] = depth_path
                tmp_dict["K"] = K
                tmp_dict["img_path"] = img_path
                # import ipdb;ipdb.set_trace()
                stereo_pkl_dict.append(tmp_dict)
                
            print('accumulate:', len(stereo_pkl_dict))
import pdb;pdb.set_trace()
with open('/cpfs01/shared/opendrivelab/opendrivelab_hdd/haoran/drivingstereo_new/OpenDataLab___DrivingStereo/drivingstereo_train.pkl', 'wb') as file:
    pickle.dump(stereo_pkl_dict, file)
