import os
import pickle
import numpy as np
from tqdm import tqdm
import cv2
import json
import math
from segment_anything.datasets.utils import *

def process_data(output_pkl, omni3d_json_path = None, pred_omni3d_json_path = None):

    with open(omni3d_json_path, 'rb') as f:
        omni3d_json = json.load(f)

    image_dict = {}
    id2path = {}
    for image in omni3d_json['images']:
        image_dict[image['file_path'].replace('//', '/')] = {}
        image_dict[image['file_path'].replace('//', '/')]['obj_list'] = []
        image_dict[image['file_path'].replace('//', '/')]['pred_obj_list'] = []
        image_dict[image['file_path'].replace('//', '/')]['K'] = image['K']
        
        id2path[image['id']] = image['file_path'].replace('//', '/')

    for anno in omni3d_json['annotations']:
        path = id2path[anno['image_id']]
        image_dict[path]['obj_list'].append(anno)
    
    if pred_omni3d_json_path is not None:
        print('this script is to process cube-rcnn output 2d results as 3daw input')
        with open(pred_omni3d_json_path, 'rb') as f:
            pred_omni3d_json = json.load(f)
        for pred_instance in pred_omni3d_json:
            path = id2path[pred_instance['image_id']]
            image_dict[path]['pred_obj_list'].append(pred_instance)
    
    path2id = {v:k for k,v in id2path.items()}

    samples = []
    for path_instance in image_dict.items():
        path, instance = path_instance
        sample = {}
        
        # img_path = './data/kitti' + path.split('KITTI_object')[-1]
        # img_path = './data/nuscenes' + path.split('nuScenes')[-1]
        img_path = './data/ARKitScenes/datasets/ARKitScenes' + path.split('ARKitScenes')[-1]
        # import ipdb; ipdb.set_trace()
        assert os.path.exists(img_path), 'img path not exist'
        todo_img = cv2.imread(img_path)
        sample['img_path'] = img_path

        depth_path = None
        sample['depth_path'] = depth_path
        
        sample['K'] = np.array(instance['K']).reshape(1, 3, 3)
        
        if pred_omni3d_json_path is not None:
            todo_obj_list = instance['pred_obj_list']
        else:
            todo_obj_list = instance['obj_list']
        if len(todo_obj_list) == 0:
            continue
        
        obj_list = list()
        
        for obj in todo_obj_list:
            x, y, z = obj['center_cam']
            w, h, l = obj['dimensions']
            if 'R_cam' in obj.keys():
                pose = np.array(obj['R_cam'])
            else:
                pose = np.array(obj['pose'])

            yaw = math.atan2(pose[0, 0], pose[2, 0])
            R_90 = np.array([
                [0,  0, 1],
                [0,  1, 0],
                [-1, 0, 0]
            ])

            pose = np.dot(pose, R_90)
            if pred_omni3d_json_path is None:
                bbox_2d_proj = obj['bbox2D_proj']
                bbox_2d_tight = obj['bbox2D_tight']
                bbox_2d_trunc = obj['bbox2D_trunc']
                label = obj['category_id']
                score = 1
                image_id = path2id[path]
                visibility = obj['visibility']
                trunction = obj['truncation']
            else:
                bbox_2d_proj = obj['bbox']
                bbox_2d_tight = [-1, -1, -1, -1]
                bbox_2d_trunc = [-1, -1, -1, -1]
                label = obj['category_id']
                score = obj['score']
                image_id = obj['image_id']
                visibility = -1
                truncation = -1

            instance_id = generate_instance_id(obj, img_path)

            obj_list.append(
                {
                    "3d_bbox": [x, y, z, w, h, l, yaw],
                    "2d_bbox_proj": bbox_2d_proj,
                    "2d_bbox_tight": bbox_2d_tight,
                    "2d_bbox_trunc": bbox_2d_trunc,
                    
                    "label": label,
                    "rotation_pose": pose,
                    "instance_id": instance_id,  # 使用字符串 ID
                    "score": score,
                    "image_id": image_id,

                    "visibility": visibility,
                    "truncation": truncation,
                }
            )

            
        #     vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, pose)
        #     K = sample['K']
        #     vertices_2d = project_to_image(vertices_3d, K.squeeze(0))
        #     fore_plane_center_2d = project_to_image(fore_plane_center_3d, K.squeeze(0))
        #     # # import ipdb;ipdb.set_trace()
        #     draw_bbox_2d(todo_img, vertices_2d)
        #     cv2.circle(todo_img, fore_plane_center_2d[0].astype(int), 2, (0, 0, 255) , 1)
        # cv2.imwrite('3D_test.png', todo_img)
        # import ipdb;ipdb.set_trace()

        sample['obj_list'] = obj_list
        if len(obj_list) > 0:
            samples.append(sample)

    print(len(samples))
    with open(output_pkl, 'wb') as f:
        pickle.dump(samples, f)

dataset_name = ['nuScenes_test', 'ARKitScenes_test', 'Hypersim_test', 'KITTI_test', 'Objectron_test', 'SUNRGBD_test']
output_pkl = './data/pkls/cubercnn2dpred/{}.pkl'.format(dataset_name[1])
omni3d_json_path = '/cpfs01/user/zhanghanxue/omni3d/datasets/Omni3D/{}.json'.format(dataset_name[1])
pred_omni3d_json_path = '/cpfs01/user/zhanghanxue/omni3d/output/evaluation/inference/iter_final/{}/omni_instances_results.json'.format(dataset_name[1])

process_data(output_pkl, omni3d_json_path, pred_omni3d_json_path)