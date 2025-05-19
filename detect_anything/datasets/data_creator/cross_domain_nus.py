import os
import pickle
import json
import math
import numpy as np
from tqdm import tqdm
from segment_anything.datasets.utils import generate_instance_id

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_image_dict(omni3d_json):
    image_dict, id2path = {}, {}
    for img in omni3d_json['images']:
        path = img['file_path'].replace('//', '/')
        image_dict[path] = {'obj_list': [], 'pred_obj_list': [], 'K': img['K']}
        id2path[img['id']] = path
    return image_dict, id2path

def process_annotations(annotations, id2path, image_dict):
    for anno in annotations:
        image_dict[id2path[anno['image_id']]]['obj_list'].append(anno)

def process_predictions(preds, id2path, image_dict):
    for inst in preds:
        image_dict[id2path[inst['image_id']]]['pred_obj_list'].append(inst)

def generate_sample(instance, path, img_path, depth_path, path2id, pred_mode):
    todo = instance['pred_obj_list'] if pred_mode else instance['obj_list']
    if not todo:
        return None

    obj_list = []
    for obj in todo:
        x,y,z = obj['center_cam']
        w,h,l = obj['dimensions']
        pose = np.array(obj.get('R_cam', obj.get('pose')))
        yaw = math.atan2(pose[0,0], pose[2,0])
        # Adjust pose
        R_90 = np.array([[0,0,1],[0,1,0],[-1,0,0]])
        pose = pose @ R_90

        if pred_mode:
            bbox2d = obj['bbox']
            score = obj['score']
            visibility = trunc = -1
        else:
            bbox2d = obj['bbox2D_proj']
            score = 1
            visibility = obj['visibility']
            trunc = obj['truncation']

        label = obj['category_id']
        inst_id = generate_instance_id(obj, img_path)

        obj_list.append({
            "3d_bbox": [x,y,z,w,h,l,yaw],
            "2d_bbox_proj": bbox2d,
            "label": label,
            "rotation_pose": pose,
            "instance_id": inst_id,
            "score": score,
            "image_id": path2id[path],
            "visibility": visibility,
            "truncation": trunc,
        })

    if not obj_list:
        return None

    return {
        "img_path": img_path,
        "depth_path": depth_path,
        "K": np.array(instance['K']).reshape(1,3,3),
        "obj_list": obj_list,
    }

def process_image_nuscenes(path, img_path, instance, path2id, pred_mode, data_root):
    # img_path = os.path.join(data_root, path)
    if not os.path.exists(img_path):
        return None
    depth_path = './data/nuscenes/nuscenes_depth/' + img_path.split('samples')[-1].replace('.jpg', '.png')
    # depth_path = os.path.join(data_root + '_depth', path.replace('.jpg', '.png'))
    if not os.path.exists(depth_path):
        depth_path = None
    return generate_sample(instance, path, img_path, depth_path, path2id, pred_mode)

def process_data_nuscenes(
    output_pkl: str,
    nuscenes_json_path: str,
    pred_nuscenes_json_path: str = None,
    data_root: str = './data/nuscenes',
    extreme_weather_pkl: str = None
):
    # 1) load omni3d json
    omni3d = load_json(nuscenes_json_path)
    img_dict, id2path = generate_image_dict(omni3d)
    process_annotations(omni3d['annotations'], id2path, img_dict)

    pred_mode = pred_nuscenes_json_path is not None
    if pred_mode:
        preds = load_json(pred_nuscenes_json_path)
        process_predictions(preds, id2path, img_dict)

    # 2) 如果给了 extreme weather pkl，就加载并提取所有 data_path
    valid_paths = None
    if extreme_weather_pkl:
        with open(extreme_weather_pkl, 'rb') as f:
            ew = pickle.load(f)
        # ew['infos'] 是一个 list，每个元素里有 'data_path'
        valid_paths = set(info['data_path'] for info in ew['infos'])
        print(f"[INFO] loaded {len(valid_paths)} valid data_paths from extreme weather info")
    # import ipdb;ipdb.set_trace()
    path2id = {v:k for k,v in id2path.items()}
    samples = []

    # 3) 逐条处理，并做过滤
    for path, inst in tqdm(img_dict.items(), desc="Processing nuScenes"):
        img_path = './data/nuscenes' + path.split('nuScenes')[-1]
        if valid_paths is not None and img_path in valid_paths:
            continue

        sample = process_image_nuscenes(path, img_path, inst, path2id, pred_mode, data_root)
        if sample:
            samples.append(sample)

    print(f"[INFO] processed {len(samples)} samples (after filtering)")
    with open(output_pkl, 'wb') as f:
        pickle.dump(samples, f)

if __name__ == "__main__":
    process_data_nuscenes(
        output_pkl="./data/pkls/nuscenes_test_filtered_indomain.pkl",
        nuscenes_json_path="/cpfs01/user/zhanghanxue/omni3d/datasets/Omni3D/nuScenes_test.json",
        pred_nuscenes_json_path=None,   # ground-truth 模式
        data_root="./data/nuscenes",
        extreme_weather_pkl="/cpfs01/user/zhanghanxue/ViDAR-private/data/nuscenes/extreme_weather_anydet3d_nuscenes_infos_val.pkl"
    )
