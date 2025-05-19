import pprint
import matplotlib.pylab as pt

import json
import os
import cv2
import numpy as np
import pickle
import copy
import concurrent.futures
from tqdm import tqdm
from os.path import join

with open('./data/A2D2/cams_lidars.json', 'r') as f:
    config = json.load(f)

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

def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q

def map_lidar_points_onto_image(image_orig, lidar, pixel_size=3, pixel_opacity=1):
    image = np.zeros_like(image_orig[:, :, 0]).astype(np.float32)
    # get rows and cols
    rows = (lidar['pcloud_attr.row'] + 0.5).astype(int)
    cols = (lidar['pcloud_attr.col'] + 0.5).astype(int)

    distances = lidar['pcloud_attr.distance']  
    # determine point colours from distance
    pixel_rowoffs = np.indices([pixel_size, pixel_size])[0] - pixel_size // 2
    pixel_coloffs = np.indices([pixel_size, pixel_size])[1] - pixel_size // 2
    canvas_rows = image.shape[0]
    canvas_cols = image.shape[1]
    for i in range(len(rows)):
        pixel_rows = np.clip(rows[i] + pixel_rowoffs, 0, canvas_rows - 1)
        pixel_cols = np.clip(cols[i] + pixel_coloffs, 0, canvas_cols - 1)
        image[pixel_rows, pixel_cols] = distances[i]

    image[image == 0.] = np.inf
    return image
def extract_image_file_name_from_lidar_file_name(file_name_lidar):
    file_name_image = file_name_lidar.split('/')
    file_name_image = file_name_image[-1].split('.')[0]
    file_name_image = file_name_image.split('_')
    file_name_image = file_name_image[0] + '_' + \
                        'camera_' + \
                        file_name_image[2] + '_' + \
                        file_name_image[3] + '.png'

    return file_name_image



def process_token(token, root_path, a2d2_pkl_dict_list, pbar):
    folder_path = f'./data/A2D2/camera_lidar/{token}/lidar/cam_front_center'
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_name_lidar = f'{token}/lidar/cam_front_center/' + file
            seq_name = file_name_lidar.split('/')[0]
            file_name_image = extract_image_file_name_from_lidar_file_name(file_name_lidar)

            instance_id = file_name_image.split('.')[0]
            file_name_image = join(root_path, seq_name, 'camera/cam_front_center/', file_name_image)
            image_front_center = cv2.imread(file_name_image)
            image_front_center = undistort_image(image_front_center, 'front_center')

            lidar_front_center = np.load(join(root_path, file_name_lidar))
            lidar_on_image = map_lidar_points_onto_image(image_front_center, lidar_front_center)

            K = np.asarray(config['cameras']['front_center']['CamMatrix']).reshape(1, 3, 3)
            # img_shape = lidar_on_image.shape
            # depth_values = lidar_on_image

            # depth_map = np.full(img_shape, 0, dtype=np.uint16)
            # max_depth = 300.0
            # depth_scaled = np.clip((depth_values / max_depth) * 65535, 0, 65535).astype(np.uint16)
            # depth_path = f'/cpfs01/shared/opendrivelab/zhanghanxue/DetAny3D/data/A2D2/depth_img/{instance_id}.png'
            depth_path = f'./data/A2D2/depth/{instance_id}.npy'
            if not os.path.exists(depth_path):
                print(f'{depth_path} not exists')
                depth_path = None

            a2d2_pkl_dict_list.append({
                'K': K,
                'img_path': file_name_image,
                'depth_path': depth_path,})

            pbar.update(1)

def main():
    a2d2_pkl_dict_list = []
    root_path = './data/A2D2/camera_lidar/'
    a2d2_bbox_token = [
        '20180810_150607', '20190401_121727', '20190401_145936'
    ]

    total_files = sum([len(files) for token in a2d2_bbox_token for root, dirs, files in os.walk(f'./data/A2D2/camera_lidar/{token}/lidar/cam_front_center')])
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        with tqdm(total=total_files, desc="Processing", unit="file") as pbar:
            for token in a2d2_bbox_token:
                futures.append(executor.submit(process_token, token, root_path, a2d2_pkl_dict_list, pbar))
            
            # 等待所有线程完成
            for future in concurrent.futures.as_completed(futures):
                future.result()

    # 将结果保存为 pickle 文件
    with open('./data/pkls/other_with_bbox_3d/A2D2_wo_box.pkl', 'wb') as f:
        pickle.dump(a2d2_pkl_dict_list, f)

if __name__ == '__main__':
    main()
