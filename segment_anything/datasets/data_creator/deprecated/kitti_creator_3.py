import os
import pickle
import numpy as np
from tqdm import tqdm
import cv2
import json
import math
from segment_anything.datasets.utils import *

def process_data(output_pkl, anydet3dpkl = None, omni3d_json_path = None, omni3d_json_path_2 = None, pred_omni3d_json_path = None, pred_omni3d_json_path_2 = None):

    with open(omni3d_json_path, 'rb') as f:
        omni3d_json = json.load(f)

    image_dict = {}
    id2path = {}
    for image in omni3d_json['images']:
        image_dict[image['file_path'].replace('//', '/')] = {}
        image_dict[image['file_path'].replace('//', '/')]['obj_list'] = []
        image_dict[image['file_path'].replace('//', '/')]['pred_obj_list'] = []
        image_dict[image['file_path'].replace('//', '/')]['K'] = image['K']
        # image_dict[image['file_path']][image['id']] = image['id']
        # import ipdb;ipdb.set_trace()
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
    
    if omni3d_json_path_2 is not None:
        with open(omni3d_json_path_2, 'rb') as f:
            omni3d_json_2 = json.load(f)
        
        for image in omni3d_json_2['images']:
            image_dict[image['file_path'].replace('//', '/')] = {}
            image_dict[image['file_path'].replace('//', '/')]['obj_list'] = []
            image_dict[image['file_path'].replace('//', '/')]['K'] = image['K']

            # image_dict[image['file_path']][image['id']] = image['id']
            # import ipdb;ipdb.set_trace()
            id2path[image['id']] = image['file_path'].replace('//', '/')
        
        for anno in omni3d_json_2['annotations']:
            path = id2path[anno['image_id']]
            image_dict[path]['obj_list'].append(anno)
    
    path2id = {v:k for k,v in id2path.items()}

    samples = []
    for path_instance in image_dict.items():
        path, instance = path_instance
        sample = {}
        img_path = '/cpfs01/shared/opendrivelab/anydet3d/data/kitti' + path.split('KITTI_object')[-1]
        assert os.path.exists(img_path), 'img path not exist'
        todo_img = cv2.imread(img_path)
        sample['img_path'] = img_path

        depth_path = None
        sample['depth_path'] = depth_path
        # import ipdb;ipdb.set_trace()
        sample['K'] = np.array(instance['K']).reshape(1, 3, 3)
        
        # omni3d_gt_centers = np.array([omni3d_gt['obj_list'][i]['center_cam'] for i in range(len(omni3d_gt['obj_list']))])
        if pred_omni3d_json_path is not None:
            todo_obj_list = instance['pred_obj_list']
        else:
            todo_obj_list = instance['obj_list']
        if len(todo_obj_list) == 0:
            # import ipdb;ipdb.set_trace()
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
            # import ipdb;ipdb.set_trace()
            R_90 = np.array([
                [0,  0, 1],
                [0,  1, 0],
                [-1, 0, 0]
            ])

            pose = np.dot(pose, R_90)
        
            if pred_omni3d_json_path is None:
                if obj['visibility'] < 0.34:
                    continue
                if obj['truncation'] > 0.33:
                    continue
                if obj['bbox2D_proj'][3] - obj['bbox2D_proj'][1] < 0.05 * todo_img.shape[0]:
                    continue
                bbox_2d = obj['bbox2D_proj']
                label = obj['category_id']
                score = 1
                image_id = path2id[path]
            else:
                bbox_2d = obj['bbox']
                label = obj['category_id']
                score = obj['score']
                image_id = obj['image_id']

            instance_id = generate_instance_id(obj, img_path)

            obj_list.append(
                {
                    "3d_bbox": [x, y, z, w, h, l, yaw],
                    "2d_bbox": bbox_2d,
                    "label": label,
                    "rotation_pose": pose,
                    "instance_id": instance_id,  # 使用字符串 ID
                    "score": score,
                    "image_id": image_id,
                }
            )

            # image = cv2.imread(img_path)
            # vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, pose)
            # K = sample['K']
            # vertices_2d = project_to_image(vertices_3d, K.squeeze(0))
            # fore_plane_center_2d = project_to_image(fore_plane_center_3d, K.squeeze(0))
            # # # import ipdb;ipdb.set_trace()
            # draw_bbox_2d(image, vertices_2d)
            # cv2.circle(image, fore_plane_center_2d[0].astype(int), 2, (0, 0, 255) , 1)
            # cv2.imwrite('3D_test.png', image)
            # import ipdb;ipdb.set_trace()

        sample['obj_list'] = obj_list
        if len(obj_list) > 0:
            samples.append(sample)
    # import ipdb;ipdb.set_trace()
    print(len(samples))
    with open(output_pkl, 'wb') as f:
        pickle.dump(samples, f)

output_pkl = '/cpfs01/shared/opendrivelab/zhanghanxue/DetAny3D/data/kitti/kitti_data_test_detany3d_according_to_omni3d_gt.pkl'
anydet3dpkl = '/cpfs01/user/zhanghanxue/segment-anything/data/kitti/kitti_data_val_detany3d_no_occlusion_v3.pkl'
omni3d_json_path = '/cpfs01/user/zhanghanxue/omni3d/datasets/Omni3D/KITTI_test.json'
# pred_omni3d_json_path = '/cpfs01/user/zhanghanxue/omni3d/output/evaluation/inference/iter_final/KITTI_test/omni_instances_results.json'
pred_omni3d_json_path = None
# omni3d_json_path_2 = '/cpfs01/user/zhanghanxue/omni3d/datasets/Omni3D/KITTI_val.json'
omni3d_json_path_2 = None
pred_omni3d_json_path_2 = None

process_data(output_pkl, anydet3dpkl, omni3d_json_path, omni3d_json_path_2, pred_omni3d_json_path, pred_omni3d_json_path_2)