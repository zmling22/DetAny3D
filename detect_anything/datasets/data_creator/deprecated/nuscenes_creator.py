import os
import pickle
import numpy as np
from tqdm import tqdm
import cv2
import json

def process_data(output_pkl, anydet3dpkl = None, omni3d_json_path = None, omni3d_json_path_2 = None):

    with open(omni3d_json_path, 'rb') as f:
        omni3d_json = json.load(f)

    image_dict = {}
    id2path = {}
    for image in omni3d_json['images']:
        image_dict[image['file_path']] = {}
        image_dict[image['file_path']]['obj_list'] = []

        # image_dict[image['file_path']][image['id']] = image['id']
        # import ipdb;ipdb.set_trace()
        id2path[image['id']] = image['file_path']

    for anno in omni3d_json['annotations']:
        path = id2path[anno['image_id']]
        image_dict[path]['obj_list'].append(anno)
    
    if omni3d_json_path_2 is not None:
        with open(omni3d_json_path_2, 'rb') as f:
            omni3d_json_2 = json.load(f)
        
        for image in omni3d_json_2['images']:
            image_dict[image['file_path']] = {}
            image_dict[image['file_path']]['obj_list'] = []

            # image_dict[image['file_path']][image['id']] = image['id']
            # import ipdb;ipdb.set_trace()
            id2path[image['id']] = image['file_path']
        
        for anno in omni3d_json_2['annotations']:
            path = id2path[anno['image_id']]
            image_dict[path]['obj_list'].append(anno)

    with open(anydet3dpkl, 'rb') as f:
        origin_dataset_pkl = pickle.load(f)
        print(len(origin_dataset_pkl['infos']))
    
    samples = []
    for instance in origin_dataset_pkl['infos']:
        sample = {}
        img_path = '/cpfs01/shared/opendrivelab/anydet3d/data' + instance['data_path'].split('/data')[-1]
        todo_img = cv2.imread(img_path)
        assert os.path.exists(img_path), 'img path not exist'
        
        sample['K'] = instance['cam_intrinsic'].reshape(1, 3, 3)
        sample['img_path'] = img_path
        depth_path = '/cpfs01/shared/opendrivelab/zhanghanxue/DetAny3D/nuscene_pre' + instance['data_path'].split('/samples')[-1].replace('.jpg', '.png')
        assert os.path.exists(depth_path), 'depth path does not exist'
        sample['depth_path'] = depth_path

        omni3d_key = 'nuScenes/samples' + instance['data_path'].split('/samples')[-1]
        if omni3d_key not in image_dict:
            print('key not found', img_path)
            continue
        omni3d_gt = image_dict[omni3d_key]
        omni3d_gt_centers = np.array([omni3d_gt['obj_list'][i]['center_cam'] for i in range(len(omni3d_gt['obj_list']))])
        if len(omni3d_gt['obj_list']) == 0:
            # import ipdb;ipdb.set_trace()
            continue
        
        obj_list = list()
        
        for i in range(len(instance['gt_boxes_3d_cam'])):
            x, y, z, l, h, w, yaw = instance['gt_boxes_3d_cam'][i]
            center_todo = np.array([x, y, z])

            distances = np.linalg.norm(omni3d_gt_centers - center_todo, axis=1)
            # print(distances)
            nearest_index = np.argmin(distances)
            nearest_point = omni3d_gt_centers[nearest_index]
            nearest_distance = distances[nearest_index]

            if nearest_distance > 0.5:
                continue

            omni3d_info = omni3d_gt['obj_list'][nearest_index]

            if omni3d_info['visibility'] < 0.34:
                continue
            if omni3d_info['truncation'] > 0.33:
                continue
            if omni3d_info['bbox2D_proj'][3] - omni3d_info['bbox2D_proj'][1] < 0.0625 * todo_img.shape[0]:
                continue

            yaw += np.pi/2

            obj_list.append(
                {
                    "3d_bbox": [x, y, z, w, h, l, yaw],
                    "2d_bbox": instance["gt_boxes_2d_img"][i],
                    "label": instance["gt_names"][i],
                }
            )

        sample['obj_list'] = obj_list
        if len(obj_list) > 0:
            samples.append(sample)
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(samples, f)

output_pkl = '/cpfs01/shared/opendrivelab/zhanghanxue/DetAny3D/data/nuscenes/nuscenes_data_train_detany3d.pkl'
anydet3dpkl = '/cpfs01/user/zhanghanxue/segment-anything/data/nuscenes/anydet3d_nuscenes_infos_train.pkl'
omni3d_json_path = '/cpfs01/user/zhanghanxue/omni3d/datasets/Omni3D/nuScenes_train.json'
omni3d_json_path_2 = '/cpfs01/user/zhanghanxue/omni3d/datasets/Omni3D/nuScenes_val.json'

process_data(output_pkl, anydet3dpkl, omni3d_json_path, omni3d_json_path_2)