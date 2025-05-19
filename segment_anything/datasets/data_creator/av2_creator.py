import os
import numpy as np
import pandas as pd
import pickle
from scipy.spatial.transform import Rotation

import cv2
from tqdm import tqdm

from segment_anything.datasets.utils import *

def quat_to_mat(quat_wxyz):
    """Convert a quaternion to a 3D rotation matrix.

    NOTE: SciPy uses the scalar last quaternion notation. Throughout this repository,
        we use the scalar FIRST convention.

    Args:
        quat_wxyz: (...,4) array of quaternions in scalar first order.

    Returns:
        (...,3,3) 3D rotation matrix.
    """
    # Convert quaternion from scalar first to scalar last.
    quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
    mat = Rotation.from_quat(quat_xyzw).as_matrix()
    return mat

class SE3:
    """SE(3) lie group object.

    References:
        http://ethaneade.com/lie_groups.pdf

    Args:
        rotation: Array of shape (3, 3)
        translation: Array of shape (3,)
    """
    
    def __init__(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        self.rotation = rotation
        self.translation = translation

    def __post_init__(self) -> None:
        """Check validity of rotation and translation arguments.

        Raises:
            ValueError: If rotation is not shape (3,3) or translation is not shape (3,).
        """
        if self.rotation.shape != (3, 3):
            raise ValueError("Rotation matrix must be shape (3,3)!")
        if self.translation.shape != (3,):
            raise ValueError("Translation vector must be shape (3,)")

    def transform_matrix(self):
        """4x4 homogeneous transformation matrix."""
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = self.rotation
        transform_matrix[:3, 3] = self.translation
        return transform_matrix

    def transform_from(self, point_cloud):
        """Apply the SE(3) transformation to this point cloud.

        Args:
            point_cloud: Array of shape (N, 3). If the transform represents dst_SE3_src,
                then point_cloud should consist of points in frame `src`

        Returns:
            Array of shape (N, 3) representing the transformed point cloud, i.e. points in frame `dst`
        """
        return point_cloud @ self.rotation.T + self.translation

    def transform_point_cloud(self, point_cloud):
        """Apply the SE(3) transformation to this point cloud.

        Args:
            point_cloud: Array of shape (N, 3). If the transform represents dst_SE3_src,
                then point_cloud should consist of points in frame `src`

        Returns:
            Array of shape (N, 3) representing the transformed point cloud, i.e. points in frame `dst`
        """
        return self.transform_from(point_cloud)

    def inverse(self):
        """Return the inverse of the current SE(3) transformation.

        For example, if the current object represents target_SE3_src, we will return instead src_SE3_target.

        Returns:
            instance of SE3 class, representing inverse of SE3 transformation target_SE3_src.
        """
        return SE3(
            rotation=self.rotation.T, translation=self.rotation.T.dot(-self.translation)
        )

    def compose(self, right_SE3):
        """Compose (right multiply) this class' transformation matrix T with another SE(3) instance.

        Algebraic representation: chained_se3 = T * right_SE3

        Args:
            right_SE3: Another instance of SE3 class.

        Returns:
            New instance of SE3 class.
        """
        chained_transform_matrix = (
            self.transform_matrix @ right_SE3.transform_matrix
        )
        chained_SE3 = SE3(
            rotation=chained_transform_matrix[:3, :3],
            translation=chained_transform_matrix[:3, 3],
        )
        return chained_SE3

def generate_pkl(data_root, pkl_path):
    data_list = []
    filenames = os.listdir(data_root)
    # number = 0
    for sequence in tqdm(filenames):
        
        sequence_path = os.path.join(data_root, sequence)
        cali_folder = os.path.join(sequence_path, 'calibration')
        intr_data = pd.read_feather(cali_folder + '/intrinsics.feather')
        extr_data = pd.read_feather(cali_folder + '/egovehicle_SE3_sensor.feather')


        

        # img_folder = os.path.join(sequence_path, 'sensors', 'cameras', 'ring_front_center')
        # number += len(os.listdir(img_folder))
        # print(len(os.listdir(img_folder)))
        # continue


        lidar_folder = os.path.join(sequence_path, 'sensors', 'lidar')
        lidar_time = []
        lidar_data = []
        for lidar in os.listdir(lidar_folder):
            lidar_path = os.path.join(lidar_folder, lidar)
            lidar_time.append(eval(lidar[:-8]))
            lidar_frame = pd.read_feather(lidar_path)
            x, y, z = lidar_frame['x'], lidar_frame['y'], lidar_frame['z']
            x, y, z = np.array(x), np.array(y), np.array(z)
            lidar_points = np.column_stack((x, y, z))
            lidar_data.append(lidar_points)
        # pdb.set_trace()
        #对于九个相机
        for index in range(9):
            #相机内参
            fx, fy, cx, cy = intr_data['fx_px'][index], intr_data['fy_px'][index], intr_data['cx_px'][index], intr_data['cy_px'][index]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            #转移矩阵
            qw, qx, qy, qz = extr_data['qw'][index], extr_data['qx'][index], extr_data['qy'][index], extr_data['qz'][index]            
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            #平移量
            tx, ty, tz = extr_data['tx_m'][index], extr_data['ty_m'][index], extr_data['tz_m'][index]
            t_vec = np.array([[tx], [ty], [tz]])
            #相机外参矩阵
            extrinsic_matrix = np.eye(4)
            extrinsic_matrix[:3, :3] = R
            extrinsic_matrix[:3, 3] = t_vec.flatten()

            # 计算旋转矩阵的逆（即转置）
            R_inv = R.T
            # 计算逆平移向量
            translation_vector_inv = -R_inv.dot(t_vec.flatten())   
            # 计算逆转换矩阵
            ego2cam = np.eye(4)
            ego2cam[:3, :3] = R_inv
            ego2cam[:3, 3] = translation_vector_inv  


            sensor_name = intr_data['sensor_name'][index]
            img_folder = os.path.join(sequence_path, 'sensors', 'cameras', sensor_name)

            for img in os.listdir(img_folder):

                img_path = os.path.join(img_folder, img)
                img_time = eval(img[:-4])
                l, r, lidar_index = 0, len(lidar_time) - 1, 0
                while l <= r:
                    mid = (l + r) // 2
                    if lidar_time[mid] < img_time:
                        l = mid + 1
                        lidar_index = mid
                    else:
                        r = mid - 1
                if lidar_index + 1 < len(lidar_time) and abs(lidar_time[lidar_index + 1] - img_time) < abs(lidar_time[lidar_index] - img_time):
                    lidar_index += 1

                lidar_points = lidar_data[lidar_index]
                depth_map, filename = generate_depth_npy(lidar_points, K, ego2cam, img_path)
                data_dict = {
                    'img_path': img_path,
                    'K': K.reshape(1, 3, 3),
                    # 'depth': depth_map,
                    'depth_path': filename
                }
                data_list.append(data_dict)
    # pdb.set_trace()
    with open(pkl_path, 'wb') as f:
        pickle.dump(data_list, f)
        
def generate_depth_map(lidar_points, P, ego2cam, img_path):
    img = cv2.imread(img_path)
    img_shape = img.shape[:2] 
    # print(lidar_points.shape)
    lidar_points_hom = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
    # print(lidar_points_hom.shape)
    points_in_cam = lidar_points_hom @ ego2cam.T
    uvz = points_in_cam[points_in_cam[:, 2] > 0, :3]
    # import pdb; pdb.set_trace()
    uvz = uvz @ P[:3, :3].T
    uvz[:, :2] /= uvz[:, 2:]
    uvz = uvz[(uvz[:, 0] >= 0) & (uvz[:, 0] < img_shape[1]) &
              (uvz[:, 1] >= 0) & (uvz[:, 1] < img_shape[0])]
    uv = uvz[:, :2]
    uv = uv.astype(int)
    depth_map = np.full(img_shape, np.inf)
    depth_map[uv[:, 1], uv[:, 0]] = uvz[:, 2]

    depth_img_path = img_path.replace('.jpg', '_depth.png')
    
    parts = depth_img_path.split('/')
    dir_depth_img_path = '/cpfs01/user/jianghaoran/detany3d/depth_images_AV2'
    # import pdb; pdb.set_trace()
    new_file_path = os.path.join(dir_depth_img_path, parts[-5], parts[-2])
    
    filename_depth = os.path.join(new_file_path, f'{parts[-1]}')
    filename_raw = os.path.join(new_file_path, f'raw_{parts[-1]}')
    cv2.imwrite(filename_raw, img)
    
    depth_img = np.zeros(img.shape)
    for u, v in uv:
        cv2.circle(depth_img, (u, v), radius=3, thickness=-1, color=(0, 255, 0))

    if not os.path.exists(new_file_path):
        os.makedirs(new_file_path)
    # import pdb; pdb.set_trace()

    cv2.imwrite(filename_depth, depth_img)
    return depth_map, filename_depth

def generate_depth_npy(lidar_points, P, ego2cam, img_path):
    img = cv2.imread(img_path)
    img_shape = img.shape[:2] 
    # print(lidar_points.shape)
    lidar_points_hom = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
    # print(lidar_points_hom.shape)
    points_in_cam = lidar_points_hom @ ego2cam.T
    uvz = points_in_cam[points_in_cam[:, 2] > 0, :3]
    # import pdb; pdb.set_trace()
    uvz = uvz @ P[:3, :3].T
    uvz[:, :2] /= uvz[:, 2:]
    uvz = uvz[(uvz[:, 0] >= 0) & (uvz[:, 0] < img_shape[1]) &
              (uvz[:, 1] >= 0) & (uvz[:, 1] < img_shape[0])]
    uv = uvz[:, :2]
    uv = uv.astype(int)
    depth_map = np.full(img_shape, np.inf)
    depth_map[uv[:, 1], uv[:, 0]] = uvz[:, 2]

    depth_npy_path = img_path.replace('.jpg', '.npy')
    
    parts = depth_npy_path.split('/')
    dir_depth_npy_path = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/GenAD_Proj/ad_datasets/Av2_sensor/val_depth'
    new_file_path = os.path.join(dir_depth_npy_path, parts[-5], parts[-2])
    
    filename = os.path.join(new_file_path, parts[-1])

    if not os.path.exists(new_file_path):
        os.makedirs(new_file_path)
    
    # pdb.set_trace()
   
    np.save(filename, depth_map)
    return depth_map, filename



def generate_pkl_front(data_root, pkl_path):
    data_list = []
    filenames = os.listdir(data_root)
    # number = 0
    for sequence in tqdm(filenames):
        
        sequence_path = os.path.join(data_root, sequence)
        cali_folder = os.path.join(sequence_path, 'calibration')
        intr_data = pd.read_feather(cali_folder + '/intrinsics.feather')
        extr_data = pd.read_feather(cali_folder + '/egovehicle_SE3_sensor.feather')
        anno_folder = os.path.join(sequence_path, 'annotations')
        annotations = pd.read_feather(anno_folder + '.feather')
        
        lidar_folder = os.path.join(sequence_path, 'sensors', 'lidar')
        lidar_time = []
        lidar_data = []
        for lidar in os.listdir(lidar_folder):
            lidar_path = os.path.join(lidar_folder, lidar)
            lidar_time.append(eval(lidar[:-8]))
            lidar_frame = pd.read_feather(lidar_path)
            x, y, z = lidar_frame['x'], lidar_frame['y'], lidar_frame['z']
            x, y, z = np.array(x), np.array(y), np.array(z)
            lidar_points = np.column_stack((x, y, z))
            lidar_data.append(lidar_points)

        #对于第一个相机
        index = 0
        #相机内参
        fx, fy, cx, cy = intr_data['fx_px'][index], intr_data['fy_px'][index], intr_data['cx_px'][index], intr_data['cy_px'][index]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        #转移矩阵
        qw, qx, qy, qz = extr_data['qw'][index], extr_data['qx'][index], extr_data['qy'][index], extr_data['qz'][index]            
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        #平移量
        tx, ty, tz = extr_data['tx_m'][index], extr_data['ty_m'][index], extr_data['tz_m'][index]
        t_vec = np.array([[tx], [ty], [tz]])
        #相机外参矩阵
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = t_vec.flatten()

        # 计算旋转矩阵的逆（即转置）
        R_inv = R.T
        # 计算逆平移向量
        translation_vector_inv = -R_inv.dot(t_vec.flatten())   
        # 计算逆转换矩阵
        ego2cam = np.eye(4)
        ego2cam[:3, :3] = R_inv
        ego2cam[:3, 3] = translation_vector_inv  


        sensor_name = intr_data['sensor_name'][index]
        img_folder = os.path.join(sequence_path, 'sensors', 'cameras', sensor_name)

        for i, img in enumerate(os.listdir(img_folder)):
            if i % 10 != 0:
                continue
            img_path = os.path.join(img_folder, img)
            img_time = eval(img[:-4])

            # 二分查找 img timestamp 最近的 lidar timestamp
            l, r, lidar_index = 0, len(lidar_time) - 1, 0
            while l <= r:
                mid = (l + r) // 2
                if lidar_time[mid] < img_time:
                    l = mid + 1
                    lidar_index = mid
                else:
                    r = mid - 1
            if lidar_index + 1 < len(lidar_time) and abs(lidar_time[lidar_index + 1] - img_time) < abs(lidar_time[lidar_index] - img_time):
                lidar_index += 1

            lidar_points = lidar_data[lidar_index]
            timestamp_ns = lidar_time[lidar_index]
            curr_annotations = annotations[annotations["timestamp_ns"] == timestamp_ns]
            curr_annotations = curr_annotations[curr_annotations["num_interior_pts"] > 2]

            raw_image = cv2.imread(img_path)

            obj_list = list()
            for annotation in curr_annotations.iterrows():
                class_name = annotation[1]["category"] 
                instance_id = annotation[1]["track_uuid"]
                num_interior_pts = annotation[1]["num_interior_pts"]

                translation = np.array([annotation[1]["tx_m"], annotation[1]["ty_m"], annotation[1]["tz_m"]])
                lwh = np.array([annotation[1]["length_m"], annotation[1]["width_m"], annotation[1]["height_m"]])
                rotation = quat_to_mat(np.array([annotation[1]["qw"], annotation[1]["qx"], annotation[1]["qy"], annotation[1]["qz"]]))
                ego_SE3_object = SE3(rotation=rotation, translation=translation)

                rot = ego_SE3_object.rotation
                lwh = lwh.tolist()
                center = translation.tolist()
                center[2] = center[2] #- lwh[2] / 2

                yaw = -math.atan2(rot[1, 0], rot[0, 0])
                pose = np.array([
                    [np.cos(yaw + np.pi), 0, np.sin(yaw + np.pi)],
                    [0, 1, 0],
                    [-np.sin(yaw + np.pi), 0, np.cos(yaw + np.pi)]
                ])
                diag_matrix = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1]
                ])

                pose = np.dot(diag_matrix, pose.T)

                # x, y, z = center @ R_inv.T
                center_hom = np.hstack((np.array([center]), np.ones((1, 1))))
                center_cam = (center_hom @ ego2cam.T).squeeze()
                x, y, z = center_cam[:3]
                (l, w, h) = lwh
                gt_bboxes_3d = [x, y, z, w, h, l, yaw]
                
                x, y, z, w, h, l, yaw = gt_bboxes_3d
                center_2d = project_to_image(np.array([[x, y, z]]), K).squeeze()
                image_h, image_w = raw_image.shape[:2]
                if center_2d[0] < 0 or center_2d[0] > image_w or center_2d[1] < 0 or center_2d[1] > image_h:
                    continue

                

                obj_list.append({
                    "3d_bbox": [x, y, z, w, h, l, yaw],
                    "2d_bbox_proj": [-1, -1, -1, -1],
                    "2d_bbox_tight": [-1, -1, -1, -1],
                    "2d_bbox_trunc": [-1, -1, -1, -1],
                    "label": class_name,
                    "rotation_pose": pose,
                    "instance_id": instance_id,
                    "score": 1,
                    "image_id": None,
                    "visibility": -1,
                    "truncation": -1,
                    "num_interior_pts": num_interior_pts,
                })

            #     vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw)
            #     vertices_2d = project_to_image(vertices_3d, K)
            #     fore_plane_center_2d = project_to_image(fore_plane_center_3d, K)
            #     # to_draw = raw_image.copy()
            #     draw_bbox_2d(raw_image, vertices_2d)
            #     cv2.circle(raw_image, fore_plane_center_2d[0].astype(int), 2, (0, 0, 255) , 1)
                
            # import ipdb; ipdb.set_trace()
            # print('stop here')
            # cv2.imwrite('3D_test_change_K.png', raw_image)
            depth_map, filename = generate_depth_npy(lidar_points, K, ego2cam, img_path)
            data_dict = {
                'img_path': img_path,
                'K': K.reshape(1, 3, 3),
                'obj_list': obj_list,
                # 'depth': depth_map,
                'depth_path': filename
            }
            data_list.append(data_dict)
    # pdb.set_trace()
    with open(pkl_path, 'wb') as f:
        pickle.dump(data_list, f)


val_data_root = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/argoverse2/sensor/val'
val_pkl_path = '/cpfs01/user/zhanghanxue/segment-anything/data/pkls/other_with_bbox_3d/Av2_val_front_with_bbox.pkl'

# /cpfs01/shared/opendrivelab/zhanghanxue/DetAny3D/<your_dataset>/depth

generate_pkl_front(val_data_root, val_pkl_path)