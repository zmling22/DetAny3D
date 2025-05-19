import os
import pickle
import numpy as np
from tqdm import tqdm
import cv2
import json
import math
from segment_anything.datasets.utils import *
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_obj_list(path_instance):
    path, instance = path_instance
    sample = {}

    img_path = './data/' + path
    if not os.path.exists(img_path):
        return None

    todo_img = cv2.imread(img_path)
    if todo_img is None:
        return None

    sample['K'] = np.array(instance['K']).reshape(1, 3, 3)
    sample['img_path'] = img_path

    depth_path = './data/hypersim/depth_in_meter/' + path.split('/')[1] + '/images/' + path.split('/')[3].replace('final_preview', 'geometry_hdf5') + '/' + path.split('/')[-1].replace('tonemap.jpg', 'depth_meters.hdf5')
    if not os.path.exists(depth_path):
        return None

    sample['depth_path'] = depth_path

    if len(instance['obj_list']) == 0:
        return None

    obj_list = []
    for obj in instance['obj_list']:
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
                "visibility": obj['visibility'],
                "truncation": obj['truncation'],
            }
        )


    sample['obj_list'] = obj_list
    return sample if len(obj_list) > 0 else None

def process_data(output_pkl, omni3d_json_path = None):

    with open(omni3d_json_path, 'rb') as f:
        omni3d_json = json.load(f)

    image_dict = {}
    id2path = {}
    for image in omni3d_json['images']:
        image_dict[image['file_path'].replace('//', '/')] = {}
        image_dict[image['file_path'].replace('//', '/')]['obj_list'] = []
        image_dict[image['file_path'].replace('//', '/')]['K'] = image['K']

        id2path[image['id']] = image['file_path'].replace('//', '/')

    for anno in omni3d_json['annotations']:
        path = id2path[anno['image_id']]
        image_dict[path]['obj_list'].append(anno)
    

    samples = []

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_obj_list, item): item for item in image_dict.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
            try:
                result = future.result()
                if result is not None:
                    samples.append(result)
            except Exception as e:
                print(f"Error processing item: {e}")

    with open(output_pkl, 'wb') as f:
        pickle.dump(samples, f)

output_pkl = './data/pkls/hypersim_test_detany3d.pkl'
omni3d_json_path = '/cpfs01/user/zhanghanxue/omni3d/datasets/Omni3D/Hypersim_test.json'

process_data(output_pkl, omni3d_json_path)