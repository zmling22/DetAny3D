import pickle
from detect_anything.datasets.utils import *
import json
import numpy as np
import cv2
import math

def mat2euler(R):

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    #singular = sy < 1e-6

    x = math.atan2(R[2, 1], R[2, 2])
    y = math.atan2(-R[2, 0], sy)
    z = math.atan2(R[1, 0], R[0, 0])

    return np.array([x, y, z])

# with open('/cpfs01/user/zhanghanxue/segment-anything/data/waymo/waymo_data_test_front.pkl', 'rb') as f:
#     dataset_pkl = pickle.load(f)


# for i in range(len(dataset_pkl['infos'])):
    
#     test_instance = dataset_pkl['infos'][i]
#     import ipdb;ipdb.set_trace()
#     for k in range(len(test_instance['gt_boxes_3d_cam']) // 4):
        
#         if (test_instance['gt_boxes_2d_img'][k] == np.array([0., 0., 0., 0.])).all() or test_instance['depth2d'][k] <= 0:
#             continue
#         x, y, z, l, h, w, yaw = test_instance['gt_boxes_3d_cam'][k]
#         K = test_instance['cam_intrinsic']
#         print(test_instance)
#         import ipdb;ipdb.set_trace()
#         to_draw = cv2.imread('/cpfs01/shared/opendrivelab/anydet3d/data' + test_instance['data_path'].split('/data')[1])
#         # to_draw = cv2.imread(test_instance['data_path'])
#         yaw += np.pi/2
#         vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw)
#         vertices_2d = project_to_image(vertices_3d, K)
#         fore_plane_center_2d = project_to_image(fore_plane_center_3d, K)
#         draw_bbox_2d(to_draw, vertices_2d)
#         cv2.circle(to_draw, fore_plane_center_2d[0].astype(int), 2, (0, 0, 255) , 1)
#         cv2.imwrite('3D_test_change_K.png', to_draw)


with open('/cpfs01/user/zhanghanxue/omni3d/datasets/Omni3D/SUNRGBD_val.json', 'rb') as f:
    dataset_json = json.load(f)

with open('/cpfs01/shared/opendrivelab/sun-rgbd/sunrgbd_for_dpt_train.pkl', 'rb') as f:
    original_pkl = pickle.load(f)

# import ipdb;ipdb.set_trace()
image_dict = {}
for image in dataset_json['images']:
    image_dict[image['id']] = {}
    image_dict[image['id']]['img_path'] = image['file_path']
    image_dict[image['id']]['obj_list'] = []

for anno in dataset_json['annotations']:
    image_dict[anno['image_id']]['obj_list'].append(anno)

img_path_2_depth_data = {}
for depth in original_pkl:
    img_path_2_depth_data[depth['img_path']] = depth
    
samples = []
for key in image_dict.keys():
    instance = image_dict[key]
    import ipdb;ipdb.set_trace()
    sample = {}
    img_path = instance['img_path']
    # img_key = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/Waymo_detany3d/kitti_format1010/validation/image_0/' + img_path.split('/')[-1]
    img_key = '/cpfs01/shared/opendrivelab/sun-rgbd/' + img_path
    depth_path = img_path_2_depth_data[img_key]['depth_path']
    K = img_path_2_depth_data[img_key]['K']

    sample['img_path'] = img_key
    sample['K'] = K
    sample['depth_path'] = depth_path
    # import ipdb;ipdb.set_trace()
    # sample['obj_occlusion'] = instance['occluded']
    obj_list = list()
    for k in range(len(instance['obj_list'])):
        obj = instance['obj_list'][k]
        if obj['visibility'] == -1:
            continue
        
        # corners = np.array(obj['bbox3D_cam'])
        # min_coords = np.min(corners, axis=0)
        # max_coords = np.max(corners, axis=0)

        # # 计算中心点坐标
        # center = (min_coords + max_coords) / 2
        center = obj['center_cam']
        x, y, z = center

        # 计算长宽高
        # dimensions = max_coords - min_coords
        dimensions = obj['dimensions']
        w, h, l = dimensions[0], dimensions[1], dimensions[2]

        # 计算yaw角
        import ipdb;ipdb.set_trace()
        pose = np.array(obj['R_cam'])
        yaw = math.atan2(pose[0, 0], pose[2, 0]) + np.pi
        # pitch, yaw, roll = mat2euler(pose)
        # yaw += np.pi / 2
        # pitch = math.asin(-pose[1, 2])
        # yaw = np.arctan2(corners[0][0] - corners[1][0], corners[0][2] - corners[1][2])

        to_draw = cv2.imread(img_key)
        # to_draw = cv2.imread(test_instance['data_path'])
        # yaw += np.pi/2
        vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw)
        vertices_2d = project_to_image(vertices_3d, K.squeeze(0))
        fore_plane_center_2d = project_to_image(fore_plane_center_3d, K.squeeze(0))
        draw_bbox_2d(to_draw, vertices_2d)
        cv2.circle(to_draw, fore_plane_center_2d[0].astype(int), 2, (0, 0, 255) , 1)
        cv2.imwrite('3D_test_change_K.png', to_draw)



        # obj_list.append(
        #     {
        #         "3d_bbox": [x, y, z, w, h, l, yaw],
        #         "2d_bbox": instance["gt_boxes_2d_img"][k],
        #         # "instance_id": obj["instanceId"],
        #         "label": instance["gt_names"][k],
        #     }
        # )

    sample['obj_list'] = obj_list
    samples.append(sample)

with open('test.pkl', 'wb') as f:
    pickle.dump(samples, f)


