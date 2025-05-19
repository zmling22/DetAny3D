import os
import pickle
import numpy as np
from tqdm import tqdm
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

def generate_depth_npy(lidar_points, P, Tr_velo_to_cam, img_path):
    img = cv2.imread(img_path)
    img_shape = img.shape[:2] 
    # print(lidar_points.shape)
    lidar_points_hom = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
    # print(lidar_points_hom.shape)
    points_in_cam = lidar_points_hom @ Tr_velo_to_cam
    uvz = points_in_cam[points_in_cam[:, 2] > 0, :3]
    uvz = uvz @ P[:3, :3].T
    uvz[:, :2] /= uvz[:, 2:]
    uvz = uvz[(uvz[:, 0] >= 0) & (uvz[:, 0] < img_shape[1]) &
              (uvz[:, 1] >= 0) & (uvz[:, 1] < img_shape[0])]
    uv = uvz[:, :2]
    uv = uv.astype(int)
    depth_map = np.full(img_shape, np.inf)
    depth_map[uv[:, 1], uv[:, 0]] = uvz[:, 2]

    depth_npy_path = img_path.replace('.png', '_depth.npy')
    
    parts = depth_npy_path.split('/')
    dir_depth_npy_path = '/cpfs01/shared/opendrivelab/anydet3d/data/kitti/test_depth_front'
    new_file_path = os.path.join(dir_depth_npy_path, parts[-2])
    
    filename = os.path.join(new_file_path, parts[-1])

    if not os.path.exists(new_file_path):
        os.makedirs(new_file_path)

    np.save(filename, depth_map)

    return depth_map, filename

def process_data(data_root, output_pkl, anydet3dpkl = None):

    with open(anydet3dpkl, 'rb') as f:
        origin_dataset_pkl = pickle.load(f)
        print(len(origin_dataset_pkl['infos']))
    
    samples = []
    for instance in origin_dataset_pkl['infos']:
        sample = {}
        img_path = '/cpfs01/shared/opendrivelab/anydet3d/data' + instance['data_path'].split('/data')[1]
        # import ipdb;ipdb.set_trace()
        lidar_path = '/cpfs01/shared/opendrivelab/anydet3d/data' + instance['pc_path'].split('data')[1]

        sample['K'] = instance['cam_intrinsic'].reshape(1, 3, 3)
        sample['img_path'] = img_path
        pc2cam = instance['pc2cam']
        lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        depth_map, depth_path = generate_depth_npy(lidar[:, :3], instance['cam_intrinsic'], pc2cam, img_path)

        sample['depth_path'] = depth_path
        # import ipdb;ipdb.set_trace()
        sample['obj_occlusion'] = instance['occluded']
        sample['obj_truncated'] = instance['truncated']
        obj_list = list()
        for i, obj_occ in enumerate(instance['occluded']):
            if obj_occ != 0 and obj_occ != 1:
                continue
            if instance['truncated'][i] > 0.33:
                continue
            if instance['gt_boxes_2d_img'][i][3] - instance['gt_boxes_2d_img'][i][1] < 0.0625 * 370:
                continue
            if instance['gt_names'][i] != 'Car':
                continue
            x, y, z, l, h, w, yaw = instance['gt_boxes_3d_cam'][i]
            yaw += np.pi/2

            obj_list.append(
                {
                    "3d_bbox": [x, y, z, w, h, l, yaw],
                    "2d_bbox": instance["gt_boxes_2d_img"][i],
                    # "instance_id": obj["instanceId"],
                    "label": instance["gt_names"][i],
                }
            )

        sample['obj_list'] = obj_list
        if len(obj_list) > 0:
            samples.append(sample)
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(samples, f)
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

data_root = '/cpfs01/shared/opendrivelab/anydet3d/data/kitti'
output_pkl = '/cpfs01/shared/opendrivelab/anydet3d/data/kitti/kitti_data_val_detany3d_car.pkl'
anydet3dpkl = '/cpfs01/shared/opendrivelab/anydet3d/data/kitti/anydet3d_kitti_infos_val.pkl'

process_data(data_root, output_pkl, anydet3dpkl)