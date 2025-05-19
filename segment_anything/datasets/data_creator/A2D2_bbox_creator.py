import json
import pprint

import cv2
import matplotlib.pylab as pt
import os
import numpy as np
import numpy.linalg as la
import pickle
import math
from segment_anything.datasets.utils import *
from os.path import join
import copy

import concurrent.futures
from tqdm import tqdm

def skew_sym_matrix(u):
    return np.array([[    0, -u[2],  u[1]], 
                     [ u[2],     0, -u[0]], 
                     [-u[1],  u[0],    0]])

def axis_angle_to_rotation_mat(axis, angle):
    return np.cos(angle) * np.eye(3) + \
        np.sin(angle) * skew_sym_matrix(axis) + \
        (1 - np.cos(angle)) * np.outer(axis, axis)

def read_bounding_boxes(file_name_bboxes):
    # open the file
    with open (file_name_bboxes, 'r') as f:
        bboxes = json.load(f)
        
    boxes = [] # a list for containing bounding boxes  
    
    for bbox in bboxes.keys():
        bbox_read = {} # a dictionary for a given bounding box
        bbox_read['class'] = bboxes[bbox]['class']
        bbox_read['truncation']= bboxes[bbox]['truncation']
        bbox_read['occlusion']= bboxes[bbox]['occlusion']
        bbox_read['alpha']= bboxes[bbox]['alpha']
        bbox_read['top'] = bboxes[bbox]['2d_bbox'][0]
        bbox_read['left'] = bboxes[bbox]['2d_bbox'][1]
        bbox_read['bottom'] = bboxes[bbox]['2d_bbox'][2]
        bbox_read['right']= bboxes[bbox]['2d_bbox'][3]
        bbox_read['center'] =  np.array(bboxes[bbox]['center'])
        bbox_read['size'] =  np.array(bboxes[bbox]['size'])
        angle = bboxes[bbox]['rot_angle']
        axis = np.array(bboxes[bbox]['axis'])
        bbox_read['angle'] = angle
        bbox_read['rotation'] = axis_angle_to_rotation_mat(axis, angle) 
        boxes.append(bbox_read)

    return boxes 

def extract_bboxes_file_name_from_image_file_name(file_name_image):
    file_name_bboxes = file_name_image.split('/')
    file_name_bboxes = file_name_bboxes[-1].split('.')[0]
    file_name_bboxes = file_name_bboxes.split('_')
    file_name_bboxes = file_name_bboxes[0] + '_' + \
                  'label3D_' + \
                  file_name_bboxes[2] + '_' + \
                  file_name_bboxes[3] + '.json'
    
    return file_name_bboxes

def extract_image_file_name_from_lidar_file_name(file_name_lidar):
    file_name_image = file_name_lidar.split('/')
    file_name_image = file_name_image[-1].split('.')[0]
    file_name_image = file_name_image.split('_')
    file_name_image = file_name_image[0] + '_' + \
                        'camera_' + \
                        file_name_image[2] + '_' + \
                        file_name_image[3] + '.png'

    return file_name_image

def extract_semantic_file_name_from_image_file_name(file_name_image):
    file_name_semantic_label = file_name_image.split('/')
    file_name_semantic_label = file_name_semantic_label[-1].split('.')[0]
    file_name_semantic_label = file_name_semantic_label.split('_')
    file_name_semantic_label = file_name_semantic_label[0] + '_' + \
                  'label_' + \
                  file_name_semantic_label[2] + '_' + \
                  file_name_semantic_label[3] + '.png'
    
    return file_name_semantic_label

def undistort_image(image, cam_name):
    if cam_name in ['front_left', 'front_center', \
                    'front_right', 'side_left', \
                    'side_right', 'rear_center']:
        # get parameters from config file
        intr_mat_undist = \
                    np.asarray(config['cameras'][cam_name]['CamMatrix'])
        intr_mat_dist = \
                    np.asarray(config['cameras'][cam_name]['CamMatrixOriginal'])
        dist_parms = \
                    np.asarray(config['cameras'][cam_name]['Distortion'])
        lens = config['cameras'][cam_name]['Lens']
        
        if (lens == 'Fisheye'):
            return cv2.fisheye.undistortImage(image, intr_mat_dist,\
                                        D=dist_parms, Knew=intr_mat_undist)
        elif (lens == 'Telecam'):
            return cv2.undistort(image, intr_mat_dist, \
                        distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image
    else:
        return image

def map_lidar_points_onto_image(image_orig, lidar, pixel_size=3, pixel_opacity=1):
    
    image = np.zeros_like(image_orig[:, :, 0]).astype(np.float32)
    # get rows and cols
    rows = (lidar['row'] + 0.5).astype(int)
    cols = (lidar['col'] + 0.5).astype(int)

    distances = lidar['distance']  
    # determine point colours from distance
    pixel_rowoffs = np.indices([pixel_size, pixel_size])[0] - pixel_size // 2
    pixel_coloffs = np.indices([pixel_size, pixel_size])[1] - pixel_size // 2
    canvas_rows = image.shape[0]
    canvas_cols = image.shape[1]
    for i in range(len(rows)):
        pixel_rows = np.clip(rows[i] + pixel_rowoffs, 0, canvas_rows - 1)
        pixel_cols = np.clip(cols[i] + pixel_coloffs, 0, canvas_cols - 1)
        image[pixel_rows, pixel_cols] = distances[i]

    # image[image == 0.] = np.inf
    return image


with open ('./data/A2D2/cams_lidars.json', 'r') as f:
    config = json.load(f)


def process_token(token, root_path, a2d2_pkl_dict_list, pbar):
    folder_path = f'./data/A2D2/camera_lidar_semantic_bboxes/{token}/lidar/cam_front_center'
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_name_lidar = f'{token}/lidar/cam_front_center/' + file
            seq_name = file_name_lidar.split('/')[0]
            file_name_image = extract_image_file_name_from_lidar_file_name(file_name_lidar)

            instance_id = file_name_image.split('.')[0]
            file_name_image = join(root_path, seq_name, 'camera/cam_front_center/', file_name_image)
            image_front_center = cv2.imread(file_name_image)
            image_front_center = undistort_image(image_front_center, 'front_center')

            file_name_semantic_label = extract_semantic_file_name_from_image_file_name(file_name_image)
            file_name_semantic_label = join(root_path, seq_name, 'label/cam_front_center/', file_name_semantic_label)
            semantic_image_front_center = cv2.imread(file_name_semantic_label)

            lidar_front_center = np.load(join(root_path, file_name_lidar))
            lidar_on_image = map_lidar_points_onto_image(image_front_center, lidar_front_center)

            file_name_bboxes = extract_bboxes_file_name_from_image_file_name(file_name_image)
            file_name_bboxes = join(root_path, seq_name, 'label3D/cam_front_center/', file_name_bboxes)
            boxes = read_bounding_boxes(file_name_bboxes)

            K = np.asarray(config['cameras']['front_center']['CamMatrix']).reshape(1, 3, 3)
            to_draw = copy.deepcopy(image_front_center)
            obj_list = list()

            for box in boxes:
                z, x, y = box['center']
                x = -x
                y = -y
                l, w, h = box['size']
                yaw = -box['angle']
                
                # 如果yaw角度大于π，调整到[-π, π]区间
                while yaw > np.pi:
                    yaw = yaw - 2 * np.pi
                while yaw < -np.pi:
                    yaw = yaw + 2 * np.pi

                pose = np.array([
                    [np.cos(yaw - np.pi), 0, np.sin(yaw - np.pi)],
                    [0, 1, 0],
                    [-np.sin(yaw - np.pi), 0, np.cos(yaw - np.pi)]
                ])
                diag_matrix = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1]
                ])

                pose = np.dot(diag_matrix, pose.T)

                bbox_2d = [box['top'], box['left'], box['bottom'], box['right']]
                
                obj_list.append({
                    "3d_bbox": [x, y, z, w, h, l, yaw],
                    "2d_bbox_proj": bbox_2d,
                    "2d_bbox_tight": [-1, -1, -1, -1],
                    "2d_bbox_trunc": [-1, -1, -1, -1],
                    "label": box['class'],
                    "rotation_pose": pose,
                    "instance_id": instance_id,
                    "score": 1,
                    "image_id": None,
                    "visibility": 1 - box['occlusion'] / 4,
                    "truncation": box['truncation'] / 3,
                })

            img_shape = lidar_on_image.shape
            depth_values = lidar_on_image
            depth_map = np.full(img_shape, 0, dtype=np.uint16)
            max_depth = 300.0
            depth_scaled = np.clip((depth_values / max_depth) * 65535, 0, 65535).astype(np.uint16)
            depth_path = f'/cpfs01/shared/opendrivelab/zhanghanxue/DetAny3D/data/A2D2/depth_img/{instance_id}.png'
            if not os.path.exists(depth_path):
                cv2.imwrite(depth_path, depth_map)

            a2d2_pkl_dict_list.append({
                'K': K,
                'img_path': file_name_image,
                'depth_path': depth_path,
                'obj_list': obj_list,})

            pbar.update(1)

def main():
    a2d2_pkl_dict_list = []
    root_path = './data/A2D2/camera_lidar_semantic_bboxes/'
    a2d2_bbox_token = [
        '20180807_145028', '20180810_142822', '20180925_101535', '20180925_112730', 
        '20180925_124435', '20180925_135056', '20181008_095521', '20181016_125231', 
        '20181107_132300', '20181107_132730', '20181107_133258', '20181108_084007', 
        '20181108_091945', '20181108_103155', '20181108_123750', '20181204_135952', 
        '20181204_154421', '20181204_170238'
    ]

    total_files = sum([len(files) for token in a2d2_bbox_token for root, dirs, files in os.walk(f'./data/A2D2/camera_lidar_semantic_bboxes/{token}/lidar/cam_front_center')])
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        with tqdm(total=total_files, desc="Processing", unit="file") as pbar:
            for token in a2d2_bbox_token:
                futures.append(executor.submit(process_token, token, root_path, a2d2_pkl_dict_list, pbar))
            
            # 等待所有线程完成
            for future in concurrent.futures.as_completed(futures):
                future.result()

    # 将结果保存为 pickle 文件
    with open('./data/pkls/other_with_bbox_3d/A2D2_with_box.pkl', 'wb') as f:
        pickle.dump(a2d2_pkl_dict_list, f)

if __name__ == '__main__':
    main()
