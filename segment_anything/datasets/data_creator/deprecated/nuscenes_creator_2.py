import os
import pickle
import numpy as np
from tqdm import tqdm
import cv2
import json
import math
from segment_anything.datasets.utils import *

def process_data(output_pkl, anydet3dpkl = None, omni3d_json_path = None, omni3d_json_path_2 = None):

    with open(omni3d_json_path, 'rb') as f:
        omni3d_json = json.load(f)

    image_dict = {}
    id2path = {}
    for image in omni3d_json['images']:
        image_dict[image['file_path'].replace('//', '/')] = {}
        image_dict[image['file_path'].replace('//', '/')]['obj_list'] = []

        # image_dict[image['file_path']][image['id']] = image['id']
        # import ipdb;ipdb.set_trace()
        id2path[image['id']] = image['file_path'].replace('//', '/')

    for anno in omni3d_json['annotations']:
        path = id2path[anno['image_id']]
        image_dict[path]['obj_list'].append(anno)
    
    if omni3d_json_path_2 is not None:
        with open(omni3d_json_path_2, 'rb') as f:
            omni3d_json_2 = json.load(f)
        
        for image in omni3d_json_2['images']:
            image_dict[image['file_path'].replace('//', '/')] = {}
            image_dict[image['file_path'].replace('//', '/')]['obj_list'] = []

            # image_dict[image['file_path']][image['id']] = image['id']
            # import ipdb;ipdb.set_trace()
            id2path[image['id']] = image['file_path'].replace('//', '/')
        
        for anno in omni3d_json_2['annotations']:
            path = id2path[anno['image_id']]
            image_dict[path]['obj_list'].append(anno)

    with open(anydet3dpkl, 'rb') as f:
        origin_dataset_pkl = pickle.load(f)
        print(len(origin_dataset_pkl))
    print(len(image_dict))
    
    samples = []
    for instance in origin_dataset_pkl:
        sample = {}
        img_path = instance['img_path']
        assert os.path.exists(img_path), 'img path not exist'
        todo_img = cv2.imread(img_path)
        
        sample['K'] = instance['K'].reshape(1, 3, 3)
        sample['img_path'] = img_path
        depth_path = instance['depth_path']
        assert os.path.exists(depth_path), 'depth path does not exist'
        sample['depth_path'] = depth_path

        omni3d_key = 'nuScenes' + instance['img_path'].split('nuscenes')[-1]
        
        if omni3d_key not in image_dict:
            import ipdb;ipdb.set_trace()
            print('key not found', img_path)
            continue
        omni3d_gt = image_dict[omni3d_key]
        # omni3d_gt_centers = np.array([omni3d_gt['obj_list'][i]['center_cam'] for i in range(len(omni3d_gt['obj_list']))])
        if len(omni3d_gt['obj_list']) == 0:
            # import ipdb;ipdb.set_trace()
            continue
        
        obj_list = list()
        
        for obj in omni3d_gt['obj_list']:
            # import ipdb;ipdb.set_trace()
            x, y, z = obj['center_cam']
            w, h, l = obj['dimensions']
            pose = np.array(obj['R_cam'])
            yaw = math.atan2(pose[0, 0], pose[2, 0])

            R_90 = np.array([
                [0,  0, 1],
                [0,  1, 0],
                [-1, 0, 0]
            ])
            pose = np.dot(pose, R_90)
        
            if obj['visibility'] != -1 and obj['visibility'] < 0.34:
                continue
            if obj['truncation'] > 0.77:
                continue
            if obj['bbox2D_proj'][3] - obj['bbox2D_proj'][1] < 0.05 * todo_img.shape[0]:
                continue
            
            instance_id = generate_instance_id(obj, img_path)

            obj_list.append(
                {
                    "3d_bbox": [x, y, z, w, h, l, yaw],
                    "2d_bbox_proj": obj['bbox2D_proj'],
                    "2d_bbox_tight": obj['bbox2D_tight'],
                    "2d_bbox_trunc": obj['bbox2D_trunc'],
                    "label": obj['category_id'],
                    "rotation_pose": pose,
                    "instance_id": instance_id,  # 使用字符串 ID
                }
            )

        sample['obj_list'] = obj_list
        if len(obj_list) > 0:
            samples.append(sample)
    # import ipdb;ipdb.set_trace()
    with open(output_pkl, 'wb') as f:
        pickle.dump(samples, f)

output_pkl = '/cpfs01/shared/opendrivelab/zhanghanxue/DetAny3D/data/nuscenes/nuscenes_data_train_detany3d_according_to_omni3d_with_edge_obj.pkl'
anydet3dpkl = '/cpfs01/shared/opendrivelab/zhanghanxue/DetAny3D/data/nuscenes/nuscenes_data_train_detany3d.pkl'
omni3d_json_path = '/cpfs01/user/zhanghanxue/omni3d/datasets/Omni3D/nuScenes_train.json'
omni3d_json_path_2 = '/cpfs01/user/zhanghanxue/omni3d/datasets/Omni3D/nuScenes_val.json'
# omni3d_json_path_2 = None

process_data(output_pkl, anydet3dpkl, omni3d_json_path, omni3d_json_path_2)