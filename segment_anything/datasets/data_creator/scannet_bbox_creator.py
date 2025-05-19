import json
import pickle
import numpy as np

file_path = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/haoran/ScanNet_formal_unzip/scannet_dataset_filtered.pkl'

with open(file_path, 'rb') as f:
    scannet_data = pickle.load(f)
for instance in scannet_data:
    scene_id = instance['img_path'].split('/')[-3]
    bbox_npy_path = f'/cpfs01/user/zhanghanxue/segment-anything/data/scannet/scannet_instance_data/{scene_id}_aligned_bbox.npy'
    bbox = np.load(bbox_npy_path)
    
    import ipdb; ipdb.set_trace()

print('debug')