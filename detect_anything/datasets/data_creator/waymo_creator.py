# import os
# import pickle
# import numpy as np
# from tqdm import tqdm
# import cv2
# from concurrent.futures import ProcessPoolExecutor, as_completed


# def generate_depth_npy(lidar_points, P, Tr_velo_to_cam, img_path):
#     img = cv2.imread(img_path)
#     img_shape = img.shape[:2] 
#     # print(lidar_points.shape)
#     lidar_points_hom = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
#     # print(lidar_points_hom.shape)
#     points_in_cam = lidar_points_hom @ Tr_velo_to_cam.T
#     uvz = points_in_cam[points_in_cam[:, 2] > 0, :3]
#     uvz = uvz @ P[:3, :3].T
#     uvz[:, :2] /= uvz[:, 2:]
#     uvz = uvz[(uvz[:, 0] >= 0) & (uvz[:, 0] < img_shape[1]) &
#               (uvz[:, 1] >= 0) & (uvz[:, 1] < img_shape[0])]
#     uv = uvz[:, :2]
#     uv = uv.astype(int)
#     depth_map = np.full(img_shape, np.inf)
#     depth_map[uv[:, 1], uv[:, 0]] = uvz[:, 2]

#     depth_npy_path = img_path.replace('.png', '_depth.npy')
    
#     parts = depth_npy_path.split('/')
#     dir_depth_npy_path = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/haoran/waymo/kitti_format/test_depth_front'
#     new_file_path = os.path.join(dir_depth_npy_path, parts[-2])
    
#     filename = os.path.join(new_file_path, parts[-1])

#     if not os.path.exists(new_file_path):
#         os.makedirs(new_file_path)
    

#     # import pdb; pdb.set_trace()
   
#     np.save(filename, depth_map)

#     return depth_map, filename

# def parse_calib_file(file):
#     with open(file, 'r') as f:
#         calib_info = {}
#         for line in f:
#             key, value = line.split(':', 1)
#             value = np.array([float(x) for x in value.split()])
            
#             if key.startswith('P'):
#                 calib_info[key] = value.reshape(3, 4)
#             elif key == 'R0_rect':
#                 calib_info['R0_rect'] = np.eye(4)
#                 calib_info['R0_rect'][:3, :3] = value.reshape(3, 3)
#             elif key.startswith('Tr_velo_to_cam'):
#                 # 动态创建不同的 Tr_velo_to_cam 矩阵
#                 calib_info[key] = np.eye(4)
#                 calib_info[key][:3, :] = value.reshape(3, 4)
    
#     return calib_info

# def process_single_image(img_file, calib_dir, lidar_dir, image_dirs):
#     samples = []
#     lidar_file = img_file.replace('.png', '.bin')
#     lidar_path = os.path.join(lidar_dir, lidar_file)

#     calib_file = os.path.join(calib_dir, img_file.replace('.png', '.txt'))
    
    
#     calib_info = parse_calib_file(calib_file)
#     lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 6)

    
#     sample = {}
#     img_path = os.path.join(image_dirs, img_file)
#     # import pdb; pdb.set_trace()
#     depth_map, depth_path = generate_depth_npy(lidar[:, :3], calib_info[f'P{0}'], calib_info[f'Tr_velo_to_cam_{0}'], img_path)

#     sample['img_path'] = img_path
#     sample['K'] = calib_info[f'P{0}'][:3, :3].reshape(1, 3, 3)
#     sample['depth_path'] = depth_path
#     sample['lidar_path'] = lidar_path
#     # import pdb; pdb.set_trace()
#     samples.append(sample)
#     return samples

# def process_data(data_root, output_pkl):
#     calib_dir = os.path.join(data_root, 'testing', 'calib')
#     image_dirs = os.path.join(data_root, 'testing', f'image_0')
#     # image_dirs = [os.path.join(data_root, 'training', f'image_{i}') for i in range(5)]
#     lidar_dir = os.path.join(data_root, 'testing', 'velodyne')
    
#     image_files = sorted(os.listdir(image_dirs))
#     # image_files = image_files[:2]
    
#     all_samples = []


#     # for img_file in tqdm(image_files):
#     #     sample =  process_single_image(img_file, calib_dir, lidar_dir, image_dirs)
#     #     all_samples.append(sample)

#     with ProcessPoolExecutor(max_workers=25) as executor:
#         futures = [executor.submit(process_single_image, img_file, calib_dir, lidar_dir, image_dirs) for img_file in image_files]
#         for future in tqdm(as_completed(futures), total=len(image_files)):
#             all_samples.extend(future.result())

#     # Uncomment the lines below to save the result in a .pkl file
#     with open(output_pkl, 'wb') as f:
#         pickle.dump(all_samples, f)
# def load_pkl(file_path):
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#     return data
# data_root = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/haoran/waymo/kitti_format'
# output_pkl = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/haoran/waymo/waymo_data_test_front.pkl'
from pathlib import Path
import os
from os.path import exists, join

from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path
from segment_anything.datasets.utils import *

# import path, sys
# folder = path.path(__file__).abspath()
# sys.path.append(folder.parent.parent)

import numpy as np
from PIL import Image
from skimage import io
import cv2


def get_image_index_str(img_idx, use_prefix_id=False):
    if use_prefix_id:
        return '{:07d}'.format(img_idx)
    else:
        return '{:06d}'.format(img_idx)


def get_kitti_info_path(idx,
                        prefix,
                        info_type='image_0',
                        file_tail='.png',
                        training=True,
                        relative_path=True,
                        exist_check=True,
                        use_prefix_id=False):
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += file_tail
    prefix = Path(prefix)
    if training:
        file_path = Path('training') / info_type / img_idx_str
    else:
        file_path = Path('testing') / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_image_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='image_0',
                   file_tail='.png',
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, info_type, file_tail, training,
                               relative_path, exist_check, use_prefix_id)


def get_label_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='label_0',
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, info_type, '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_plane_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='planes',
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, info_type, '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_velodyne_path(idx,
                      prefix,
                      training=True,
                      relative_path=True,
                      exist_check=True,
                      use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'velodyne', '.bin', training,
                               relative_path, exist_check, use_prefix_id)


def get_calib_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'calib', '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_pose_path(idx,
                  prefix,
                  training=True,
                  relative_path=True,
                  exist_check=True,
                  use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'pose', '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_timestamp_path(idx,
                       prefix,
                       training=True,
                       relative_path=True,
                       exist_check=True,
                       use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'timestamp', '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                    for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14])
                                          for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_kitti_image_info(path,
                         training=True,
                         label_info=True,
                         velodyne=False,
                         calib=False,
                         with_plane=False,
                         image_ids=7481,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True):
    """
    KITTI annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    # import ipdb;ipdb.set_trace()
    def map_func(idx):
        info = {}
        info['data_path'] = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/haoran/waymo/kitti_format/' + get_image_path(idx, path, training,
                                                  relative_path, exist_check=False, use_prefix_id=True)
        info['image_shape']=cv2.imread(info['data_path']).shape[:2]    
        calib_info = {}
        
        annotations = None
    
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path, exist_check=False, use_prefix_id=True)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
            # print(annotations)
        if velodyne:
            velodyne_path = get_velodyne_path(
                idx, path, training, relative_path=False)
            info['pc_path'] = velodyne_path
            info['pc_read_func_name'] = 'kitti_pc_loader'
        if calib:
            calib_path = get_calib_path(
                idx, path, training, relative_path=False, exist_check=False, use_prefix_id=True)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                           ]).reshape([3, 4])[:3,:3]
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                           ]).reshape([3, 4])[:3,:3]
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                           ]).reshape([3, 4])[:3,:3]
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                           ]).reshape([3, 4])[:3,:3]
            
            

            R0_rect = np.array([
                float(info) for info in lines[4].split(' ')[1:10]
            ]).reshape([3, 3])
            info['cam_intrinsic'] = P0
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect

                P2_4x4 = np.zeros([4, 4], dtype=P2.dtype)
                P2_4x4[3, 3] = 1.
                P2_4x4[:3, :3] = P0


            else:
                rect_4x4 = R0_rect

            Tr_velo_to_cam = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])
            
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
        
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam


            if velodyne:
                info['pc2cam'] = (rect_4x4 @ Tr_velo_to_cam).T
        
        if annotations is not None:
            # import ipdb;ipdb.set_trace()
            info['gt_boxes_2d_img'] = annotations['bbox']
            l = annotations['dimensions'][:,0].reshape([-1,1])
            h = annotations['dimensions'][:,1].reshape([-1,1])
            w = annotations['dimensions'][:,2].reshape([-1,1])
            x = annotations['location'][:,0].reshape([-1,1]) 
            y = annotations['location'][:,1].reshape([-1,1])
            z = annotations['location'][:,2].reshape([-1,1]) 

            info['gt_boxes_3d_cam'] = np.concatenate((x, y-h/2, z, l, h, w), axis=1)
            info['gt_boxes_3d_cam'] = np.concatenate((info['gt_boxes_3d_cam'], annotations['rotation_y'].reshape([len(annotations['rotation_y']),1])),axis=1)
            info['center2d'] = np.concatenate((((annotations['bbox'][:,0] + annotations['bbox'][:,2])/2).reshape([len(annotations['bbox']),1]), ((annotations['bbox'][:,1] + annotations['bbox'][:,3])/2).reshape([len(annotations['bbox']),1])),axis=1)
            info['depth2d'] = annotations['location'][:, 2]
            info['gt_names'] = annotations['name']

           
        return info
    
    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)

    return list(image_infos)



def kitti_anno_to_label_file(annos, folder):
    folder = Path(folder)
    for anno in annos:
        image_idx = anno['metadata']['image_idx']
        label_lines = []
        for j in range(anno['bbox'].shape[0]):
            label_dict = {
                'name': anno['name'][j],
                'alpha': anno['alpha'][j],
                'bbox': anno['bbox'][j],
                'location': anno['location'][j],
                'dimensions': anno['dimensions'][j],
                'rotation_y': anno['rotation_y'][j],
                'score': anno['score'][j],
            }
            label_line = kitti_result_line(label_dict)
            label_lines.append(label_line)
        label_file = folder / f'{get_image_index_str(image_idx)}.txt'
        label_str = '\n'.join(label_lines)
        with open(label_file, 'w') as f:
            f.write(label_str)


def add_difficulty_to_annos(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=bool)
    moderate_mask = np.ones((len(dims), ), dtype=bool)
    hard_mask = np.ones((len(dims), ), dtype=bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos['difficulty'] = np.array(diff, np.int32)
    return diff


def kitti_result_line(result_dict, precision=4):
    prec_float = '{' + ':.{}f'.format(precision) + '}'
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError('you must specify a value for {}'.format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError('unknown key. supported key:{}'.format(
                res_dict.keys()))
    return ' '.join(res_line)

def create_ImageSets_img_ids(root_dir, splits):
    """Create txt files indicating what to collect in each split."""
    save_dir = join(root_dir, 'ImageSets/')
    if not exists(save_dir):
        os.mkdir(save_dir)

    idx_all = [[] for _ in splits]
    for i, split in enumerate(splits):
        import ipdb; ipdb.set_trace()
        path = join(root_dir, split, 'label_0')
        if not exists(path):
            RawNames = []
        else:
            RawNames = os.listdir(path)

        for name in RawNames:
            if name.endswith('.txt'):
                idx = name.replace('.txt', '\n')
                idx_all[int(idx[0])].append(idx)
        idx_all[i].sort()
    
    open(save_dir + 'train.txt', 'w').writelines(idx_all[0])
    open(save_dir + 'val.txt', 'w').writelines(idx_all[1])
    open(save_dir + 'trainval.txt', 'w').writelines(idx_all[0] + idx_all[1])
    if len(idx_all) >= 3:
        open(save_dir + 'test.txt', 'w').writelines(idx_all[2])
    if len(idx_all) >= 4:
        open(save_dir + 'test_cam_only.txt', 'w').writelines(idx_all[3])
    print('created txt files indicating what to collect in ', splits)

# create_ImageSets_img_ids('/cpfs01/shared/opendrivelab/opendrivelab_hdd/haoran/waymo/kitti_format', ['training', 'validation', 'testing'])
def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

# import mmengine

# data_path = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/haoran/waymo/kitti_format'
# imageset_folder = Path(data_path) / 'ImageSets'
# train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
# val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
# # test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))
# save_path = Path(data_path)

# kitti_infos_train = get_kitti_image_info(
#         data_path,
#         training=True,
#         velodyne=False,
#         calib=True,
#         with_plane=False,
#         image_ids=train_img_ids,
#         relative_path=True)

# filename = save_path / 'waymo_infos_bbox_train_front.pkl'
# print(f'Waymo info test file is saved to {filename}')
# metadata = dict(version=None)
# data = dict(infos=kitti_infos_train, metadata=metadata)
# mmengine.dump(data, filename)
import pickle
import numpy as np
def process_data(output_pkl, anydet3dpkl = None, depth_pkl = None):

    with open(anydet3dpkl, 'rb') as f:
        origin_dataset_pkl = pickle.load(f)
        print(len(origin_dataset_pkl['infos']))
    with open(depth_pkl, 'rb') as f:
        depth_data = pickle.load(f)
        print(len(origin_dataset_pkl['infos']))
    
    img_path_2_depth_data = {}
    for depth in depth_data:
        key_new = depth['img_path'].split('/')[-1]
        img_path_2_depth_data[key_new] = depth
    # import ipdb;ipdb.set_trace()
    samples = []
    for instance in origin_dataset_pkl['infos']:
        # import ipdb;ipdb.set_trace()
        sample = {}
        img_path = instance['data_path']
        # img_key = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/Waymo_detany3d/kitti_format1010/validation/image_0/' + img_path.split('/')[-1]
        img_key = img_path.split('/')[-1]
        depth_path = img_path_2_depth_data[img_key]['depth_path']
        K = img_path_2_depth_data[img_key]['K']

        sample['img_path'] = img_path
        sample['K'] = K
        sample['depth_path'] = depth_path
        raw_image = cv2.imread(img_path)
        # import ipdb;ipdb.set_trace()
        # sample['obj_occlusion'] = instance['occluded']
        obj_list = list()
        for k in range(len(instance['gt_boxes_3d_cam'])):
            if (instance['gt_boxes_2d_img'][k] == np.array([0., 0., 0., 0.])).all() or instance['depth2d'][k] <= 0:
                continue
            # import ipdb;ipdb.set_trace()
            x, y, z, l, h, w, yaw = instance['gt_boxes_3d_cam'][k]
            yaw += np.pi/2
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

            categories_mapping = {
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
            }

            obj_list.append(
                {
                    "3d_bbox": [x, y, z, w, h, l, yaw],
                    "2d_bbox_proj": instance["gt_boxes_2d_img"][k].tolist(),
                    "2d_bbox_tight": [-1, -1, -1, -1],
                    "2d_bbox_trunc": [-1, -1, -1, -1],
                    "instance_id": None,
                    "label": categories_mapping[instance["gt_names"][k].lower()],
                    "rotation_pose": pose,
                    "score": 1,
                    "image_id": len(samples)+1,
                    "visibility": -1,
                    "truncation": -1,

                }
            )
            # import ipdb; ipdb.set_trace()

        #     vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, pose)
        #     vertices_2d = project_to_image(vertices_3d, K.squeeze())
        #     fore_plane_center_2d = project_to_image(fore_plane_center_3d, K.squeeze())
        #     # to_draw = raw_image.copy()
        #     draw_bbox_2d(raw_image, vertices_2d)
        #     cv2.circle(raw_image, fore_plane_center_2d[0].astype(int), 2, (0, 0, 255) , 1)
            
        # import ipdb; ipdb.set_trace()
        # print('stop here')
        # cv2.imwrite('3D_test_change_K.png', raw_image)
        # print(instance['gt_names'])
        sample['obj_list'] = obj_list
        samples.append(sample)
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(samples, f)
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

output_pkl = '/cpfs01/user/zhanghanxue/segment-anything/data/pkls/other_with_bbox_3d/waymo_data_val.pkl'
anydet3dpkl = '/cpfs01/user/zhanghanxue/segment-anything/data/waymo/kitti_format/waymo_infos_bbox_val_front.pkl'
depth_pkl = '/cpfs01/user/zhanghanxue/segment-anything/data/waymo/waymo_data_val_front.pkl'

process_data(output_pkl, anydet3dpkl, depth_pkl)


