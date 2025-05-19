import pickle
import cv2
import numpy as np
import os
import json
from pyquaternion import Quaternion

from segment_anything.datasets.utils import *

def get_K_multiplier():
    K_multiplier = np.zeros((3, 3))
    K_multiplier[0][1] = K_multiplier[1][2] = -1
    K_multiplier[2][0] = 1
    return K_multiplier


cityscapes_pkl_dict = []
for home, dirs, files in os.walk('/cpfs01/shared/opendrivelab/datasets/cityscapes3d/cityscapes3d/disparity/val'):
    print(len(files))
    print(files)
    for file in files:
        tmp_dict = {}
        city = file.split('_')[0]
        city_id = file.split('disparity')[0]
        
        img_disp_name = f'/cpfs01/shared/opendrivelab/datasets/cityscapes3d/cityscapes3d/disparity/val/{city}/{city_id}disparity.png'
        img_path = f'/cpfs01/shared/opendrivelab/datasets/cityscapes3d/cityscapes3d/leftImg8bit/val/{city}/{city_id}leftImg8bit.png'
        gt_bbox_path = f'/cpfs01/shared/opendrivelab/datasets/cityscapes3d/cityscapes3d/gtBbox3d/val/{city}/{city_id}gtBbox3d.json'
        gt_segment_mask_path = f'/cpfs01/shared/opendrivelab/datasets/cityscapes3d/cityscapes3d/gtFine/val/{city}/{city_id}gtFine_instanceIds.png'

        # img_d = cv2.imread(img_disp_name, cv2.IMREAD_UNCHANGED).astype(np.float32)
        # img_d[img_d > 0] = (img_d[img_d > 0] - 1) / 256
        # print(img_d.shape)

        with open (gt_bbox_path, 'rb') as file:
            anno = json.load(file)

        baseline = anno["sensor"]["baseline"]
        fx = anno["sensor"]["fx"]
        fy = anno["sensor"]["fy"]
        u0 = anno["sensor"]["u0"]
        v0 = anno["sensor"]["v0"]
        sensor_T_ISO_8855 = anno["sensor"]["sensor_T_ISO_8855"]
        sensor_T_ISO_8855_4x4 = np.eye(4)
        sensor_T_ISO_8855_4x4[:3, :] = np.array(sensor_T_ISO_8855)
        objects = anno["objects"]

        # depth = (baseline * fx) / img_d
        depth_path = f'/cpfs01/shared/opendrivelab/datasets/cityscapes3d/cityscapes3d/depth/val/{city}/'
        # if not os.path.isdir(depth_path):
        #     os.makedirs(depth_path)
        # np.save(f'/cpfs01/shared/opendrivelab/cityscapes3d/depth/val/{city}/{city_id}depth.npy', depth) 
        K = np.array([[[fx, 0, u0], [0, fy, v0], [0, 0, 1]]])
        tmp_dict = {}
        obj_list = []
        for obj in objects:
            
            center = np.array(obj["3d"]["center"])
            center_T = np.ones((4, 1))
            center_T[:3, 0] = center.T
            center = np.matmul(sensor_T_ISO_8855_4x4, center_T)
            center = (center.T)[0, :3]
            x, y, z = -center[1], -center[2], center[0]
            l, w, h = obj["3d"]["dimensions"]
            K_multiplier = get_K_multiplier()
            quaternion_rot = Quaternion(obj["3d"]["rotation"])
            # image_T_sensor_quaternion = Quaternion(matrix=K_multiplier)
            # quaternion_rot = (
            #     image_T_sensor_quaternion *
            #     quaternion_rot *
            #     image_T_sensor_quaternion.inverse
            # )
            rotation_matrix = np.array(quaternion_rot.rotation_matrix)
            rotation_matrix = np.matmul(
                np.matmul(K_multiplier, rotation_matrix), K_multiplier.T
            )
            r00 = rotation_matrix[0, 0]
            r02 = rotation_matrix[0, 2]

            # 计算绕 y 轴的旋转角度
            yaw = np.arctan2(r02, r00)

            # # 确保 yaw 角在 [-π, π] 范围内
            if yaw > np.pi:
                yaw -= 2 * np.pi
            if yaw < -np.pi:
                yaw += 2 * np.pi

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
            # print(yaw)
            categories2id = {
                'pedestrian': 0,
                'car': 1,
                'dontcare': 2,
                'cyclist': 3,
                'truck': 5,
                'tram': 6,
                'traffic cone': 8,
                'barrier': 9,
                'motorcycle': 10,
                'bicycle': 11,
                'bus': 12,
                'trailer': 13,
                'train': 6,
                'caravan': 1
            }
            obj_list.append(
                {
                    "3d_bbox": [x, y, z, w, h, l, yaw],
                    "2d_bbox_proj": [-1, -1, -1, -1],
                    "2d_bbox_tight": obj["2d"]["modal"],
                    "2d_bbox_trunc": [-1, -1, -1, -1],
                    "instance_id": obj["instanceId"],
                    "label": categories2id[obj["label"]],
                    "rotation_pose": pose,
                    "score": 1,
                    "image_id": len(cityscapes_pkl_dict)+1,
                    "visibility": -1,
                    "truncation": -1,
                }
            )

            # image = cv2.imread(img_path)
            # vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, pose)

            # vertices_2d = project_to_image(vertices_3d, K.squeeze(0))
            # fore_plane_center_2d = project_to_image(fore_plane_center_3d, K.squeeze(0))
            # # import ipdb;ipdb.set_trace()
            # draw_bbox_2d(image, vertices_2d)
            # cv2.circle(image, fore_plane_center_2d[0].astype(int), 2, (0, 0, 255) , 1)
            # cv2.imwrite('3D_test.png', image)
            # import ipdb;ipdb.set_trace()

        tmp_dict['depth_path'] = f'/cpfs01/shared/opendrivelab/datasets/cityscapes3d/cityscapes3d/depth/val/{city}/{city_id}depth.npy'
        tmp_dict["K"] = K
        tmp_dict["img_path"] = img_path
        tmp_dict["obj_list"] = obj_list
        tmp_dict["instance_masks"] = gt_segment_mask_path
        cityscapes_pkl_dict.append(tmp_dict)
        print('accumulate:', len(cityscapes_pkl_dict))
# with open('/cpfs01/shared/opendrivelab/datasets/cityscapes3d/cityscapes3d/cityscapes_val_with_3d_bbox_image_level.pkl', 'wb') as file:
#     pickle.dump(cityscapes_pkl_dict, file)
output_pkl = '/cpfs01/user/zhanghanxue/segment-anything/data/pkls/other_with_bbox_3d/cityscapes_val_with_box.pkl'
with open(output_pkl, 'wb') as file:
    pickle.dump(cityscapes_pkl_dict, file)
