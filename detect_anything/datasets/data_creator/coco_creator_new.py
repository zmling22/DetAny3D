from segment_anything.datasets.coco_utils import COCO
import pickle
import numpy as np
import skimage.io as io
import cv2
import torch
from segment_anything.utils.amg import batched_mask_to_box
from segment_anything import SamPredictor, sam_model_registry
from tqdm import *
import copy

dataDir='./data/coco'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

coco_for_sam = list()

coco=COCO(annFile)
data_list = coco.getImgIds()

for imgIds in tqdm(data_list): 
    import ipdb;ipdb.set_trace()
    img_info = coco.loadImgs(imgIds)[0]
    annIds = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    if len(anns) == 0:
        continue
    img_file_name = img_info['file_name']
    img_path='{}/{}/{}'.format(dataDir, dataType, img_file_name)
    img = cv2.imread(img_path)
    img_shape = img.shape
    K = np.array([[[2 * img_shape[0], 0, img_shape[1] / 2],
        [0, 2 * img_shape[0], img_shape[0] / 2],
        [0, 0, 1]]]).astype(np.float32)
    depth_path = None
    obj_list = []
    for ann in anns:
        if ann['iscrowd']:
            continue
        bbox_tight = ann['bbox']
        bbox_tight[2] = bbox_tight[2] + bbox_tight[0]
        bbox_tight[3] = bbox_tight[3] + bbox_tight[1]
        instance_id = 'coco' + str(ann['id'])

        obj_list.append({
                    "3d_bbox": [-1, -1, -1, -1, -1, -1, -1],
                    "2d_bbox_proj": [-1, -1, -1, -1],
                    "2d_bbox_tight": bbox_tight,
                    "2d_bbox_trunc": [-1, -1, -1, -1],
                    "label": ann['category_id'],
                    "rotation_pose": None,
                    "instance_id": instance_id,
                    "score": 1,
                    "image_id": None,
                    "visibility": -1,
                    "truncation": -1,
                    "rle_mask": ann['segmentation']
                })
    coco_for_sam.append(
        dict(
            K = K,
            # anns_all_img = anns,
            obj_list = obj_list,
            img_path = img_path,
            depth_path = depth_path,
            imgIds = imgIds,
            )
        )

with open('./data/pkls/coco_val.pkl', 'wb') as file:
    
    pickle.dump(coco_for_sam, file)

