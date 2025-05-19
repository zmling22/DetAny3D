import pickle
import cv2

import json
import numpy as np

from segment_anything.datasets.utils import *
from shapely.geometry import MultiPoint
from shapely.geometry import box


# with open('/cpfs01/user/zhanghanxue/segment-anything/data/pkls/other_with_bbox_3d/waymo_data_val.pkl', 'rb') as f:
#     detany3d_data = pickle.load(f)


# omni3d_data = {
#     'info': {
#         'id': 18,
#         'source': 'Waymo',
#         'name': 'Waymo Test',
#         'split': 'Test',
#         'version': '0.1',
#         'url': 'http://www.cvlibs.net/datasets/kitti/eval_object.php'
#     },
#     'images': [],
#     'categories': [],
#     'annotations': []
# }


with open('/cpfs01/user/zhanghanxue/segment-anything/data/pkls/nuscenes_test_filtered_indomain.pkl', 'rb') as f:
    detany3d_data = pickle.load(f)


omni3d_data = {
    'info': {
        'id': 19,
        'source': 'Cityscapes3D',
        'name': 'Cityscapes3D Test',
        'split': 'Test',
        'version': '0.1',
        'url': 'http://www.cvlibs.net/datasets/kitti/eval_object.php'
    },
    'images': [],
    'categories': [],
    'annotations': []
}


categories = set()  # 用于存储唯一类别
for instance in detany3d_data:
    for obj in instance['obj_list']:
        categories.add(obj['label'])

print(categories)

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
}
id2categories = {
    0: 'pedestrian',
    1: 'car',
    2: 'dontcare',
    3: 'cyclist',
    5: 'truck',
    6: 'tram',
    8: 'traffic cone',
    9: 'barrier',
    10: 'motorcycle',
    11: 'bicycle',
    12: 'bus',
    13: 'trailer',
}


# 现在我们将这些类别添加到COCO格式中
for i, category in enumerate(categories):
    # print(category)
    # waymo
    omni3d_data['categories'].append({
        'id': category,  # COCO格式中的类别ID从1开始
        'name': id2categories[category],
        'supercategory': "vehicle & road",  # 在此假设supercategory和name相同
    })
    # import ipdb;ipdb.set_trace()
    # omni3d_data['categories'].append({
    #     'id': categories2id[category],  # COCO格式中的类别ID从1开始
    #     'name': id2categories[categories2id[category]],
    #     'supercategory': "vehicle & road",  # 在此假设supercategory和name相同
    # })


for idx, instance in enumerate(detany3d_data):
    raw_img = cv2.imread(instance['img_path'])
    # import ipdb;ipdb.set_trace()
    # img_path = 'waymo' + instance['img_path'].split('waymo')[-1]
    # img_path = 'cityscapes3d' + instance['img_path'].split('cityscapes3d')[-1]
    img_path = 'nuScenes' + instance['img_path'].split('nuscenes')[-1]
    img_id = idx + 1  # 为每张图片分配一个唯一ID
    image_info = {
        'id': img_id,
        'file_path': img_path,
        'width': raw_img.shape[1],  # 根据实际图片尺寸调整
        'height': raw_img.shape[0],  # 根据实际图片尺寸调整
        'K': instance['K'][0].tolist(),  # 转换为列表格式
        'src_90_rotate': 0,
        'src_flagged': False,
        'incomplete': False,
        'dataset_id': 19  # 设置数据集ID
    }
    omni3d_data['images'].append(image_info)

# import ipdb;ipdb.set_trace()

annotation_id = 1
for id, instance in enumerate(detany3d_data):
    for obj in instance['obj_list']:
        x, y, z, l, h, w, yaw = obj['3d_bbox']

        # yaw += np.pi / 2
    
        pose = np.array([
            [np.cos(yaw + np.pi), 0, np.sin(yaw + np.pi)],
            [0, 1, 0],
            [-np.sin(yaw + np.pi), 0, np.cos(yaw + np.pi)]
        ])
        gt_3d_corners, _ = compute_3d_bbox_vertices(x, y, z, l, h, w, yaw)
        new_order = np.array([5, 1, 0, 4, 6, 2, 3, 7])

        gt_3d_corners_for_cubercnn = gt_3d_corners[new_order, :].tolist()
        if obj['2d_bbox_proj'] == [-1, -1, -1, -1]:
            vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, pose)
            vertices_2d = project_to_image(vertices_3d, instance['K'].squeeze(0))
            polygon_from_2d_box = MultiPoint(vertices_2d).convex_hull  

            img_canvas = box(0, 0, raw_img.shape[1], raw_img.shape[0]) 
            if polygon_from_2d_box.intersects(img_canvas):  
                img_intersection = polygon_from_2d_box.intersection(img_canvas)
                intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])
                min_x = min(intersection_coords[:, 0])
                min_y = min(intersection_coords[:, 1])
                max_x = max(intersection_coords[:, 0])
                max_y = max(intersection_coords[:, 1])
            else:
                # print('a potential risk of out bbox')
                continue

            bbox_2d_polygon = [min_x, min_y, max_x, max_y]
            obj['2d_bbox_proj'] = bbox_2d_polygon
            # bbox_2d_tensor = torch.tensor(bbox_2d_polygon, dtype=torch.int)

        
        # import ipdb;ipdb.set_trace()
        annotation = {
            'id': annotation_id,
            'image_id': id + 1,  # 这里用图像的路径作为 ID
            'dataset_id': 19,
            'category_id': obj['label'],  # 使用类别 ID
            'category_name': id2categories[obj['label']],
    
            'bbox2D_proj':  obj['2d_bbox_proj'] if obj.get('2d_bbox_proj') else [-1, -1, -1, -1],
            'bbox2D_tight': obj['2d_bbox_tight'] if obj.get('2d_bbox_tight') else [-1, -1, -1, -1],
            'bbox2D_trunc': obj['2d_bbox_trunc'] if obj.get('2d_bbox_trunc') else [-1, -1, -1, -1],
            'center_cam': obj['3d_bbox'][:3],  # 提取3D框中的位置信息
            'dimensions': obj['3d_bbox'][3:6],  # 提取3D框的尺寸
            'bbox3D_cam': gt_3d_corners_for_cubercnn,
            'R_cam': pose.tolist(),
            'visibility': obj['visibility'],
            'truncation': obj['truncation'],
            'behind_camera': False,
            'depth_error': -1,
            'segmentation_pts': -1,
            'lidar_pts': -1,
            'valid3D': True
        }
        annotation_id +=1
        omni3d_data['annotations'].append(annotation)

with open('/cpfs01/user/zhanghanxue/omni3d/datasets/Omni3D/nuScenes_test_filtered_indomain.json', 'w') as f:
    json.dump(omni3d_data, f, indent=4)