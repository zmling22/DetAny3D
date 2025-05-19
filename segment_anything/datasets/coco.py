from torch.utils.data import Dataset, DataLoader
import torch
from segment_anything.datasets.coco_utils import COCO
from segment_anything.datasets import transforms_shir as transforms
import torch.nn.functional as F
import cv2
import pickle
import numpy as np
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.amg import batched_mask_to_box
from segment_anything.mylogger import *
from copy import deepcopy
from segment_anything.datasets.utils import *

def init_point_sampling(mask, get_point=1):
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    # Get coordinates of black/white pixels
    mask_1 = deepcopy(mask[1:, :])
    mask_2 = deepcopy(mask[:-1, :])
    mask_3 = deepcopy(mask[:, 1:])
    mask_4 = deepcopy(mask[:, :-1])
    mask[:-1, :] += mask_1
    mask[1:, :] += mask_2
    mask[:, :-1] += mask_3
    mask[:, 1:] += mask_4

    fg_coords = np.argwhere(mask == 5)[:, ::-1]
    bg_coords = np.argwhere(mask == 0)[:, ::-1]
    fg_size = len(fg_coords)
    bg_size = len(bg_coords)
    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices],
                                                                                              dtype=torch.int)
        return coords, labels

class MyCOCODataset(Dataset):

    def __init__(self, 
                 cfg,
                 transform,
                 mode,
                 pixel_mean = [123.675, 116.28, 103.53],
                 pixel_std = [58.395, 57.12, 57.375],
                 sample_point_num = 1):
        self.sam_trans = ResizeLongestSide(cfg.model.pad)
        self.transform = transform
        if mode == 'train':
            self.dataType = cfg.dataset.train.dataType
            self.dataDir = cfg.dataset.train.dataDir
            self.pkl_path = cfg.dataset.train.pkl_path
            with open(self.pkl_path, 'rb') as f:
                self.coco_pkl = pickle.load(f)[:cfg.dataset.train.end_index]
        else:
            self.dataType = cfg.dataset.val.dataType
            self.dataDir = cfg.dataset.val.dataDir
            self.pkl_path = cfg.dataset.val.pkl_path
            with open(self.pkl_path, 'rb') as f:
                self.coco_pkl = pickle.load(f)[:cfg.dataset.val.end_index]
        self.annFile = '{}/annotations/instances_{}.json'.format(self.dataDir, self.dataType)
        self.coco = COCO(self.annFile)
        self.data_list = self.coco.getImgIds()
        self.sample_point_num = sample_point_num
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        self.target_size = cfg.model.pad
        self.sam_trans = ResizeLongestSide(self.target_size)
    def __getitem__(self, index):
        
        # imgIds = self.data_list[index]
        # img_info = self.coco.loadImgs(imgIds)[0]
        # annIds = self.coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
        # anns = self.coco.loadAnns(annIds)
        # mask = coco.annToMask(anns[0])
        # img_file_name = img_info['file_name']
        # img_path='{}/{}/{}'.format(self.dataDir, self.dataType, img_file_name)
        # todo_img = cv2.imread(img_path)
        
        ann = self.coco_pkl[index]['ann']
        anns = self.coco_pkl[index]['anns_all_img']
        mask = self.coco.annToMask(ann)
        mask_all_image = self.coco.annlistToMask(anns)

        # print(mask_all_image)

        # mask = self.coco_pkl[index]['mask']
        img_path = self.coco_pkl[index]['img_path']
        todo_img = cv2.imread(img_path)
        todo_img = cv2.cvtColor(todo_img, cv2.COLOR_BGR2RGB)
        todo_img, mask = self.transform(todo_img, mask)
        original_size = tuple(todo_img.shape[1:3])

        todo_img = todo_img.unsqueeze(0)
        todo_img = self.sam_trans.apply_image_torch(todo_img)
        before_pad_size = tuple(todo_img.shape[-2:])
        todo_img = self.preprocess(todo_img).squeeze(0)

        # import ipdb;ipdb.set_trace()
        # todo_img, mask = self.sam_trans.apply_image_torch(todo_img), self.sam_trans.apply_image_torch(mask)
        # mask[mask > 0.5] = 1
        # mask[mask <= 0.5] = 0
        # image_size = tuple(todo_img.shape[1:3])
        # todo_img, mask = self.sam_trans.preprocess(todo_img), self.sam_trans.preprocess(mask)
        point_coords, point_labels = init_point_sampling(mask, self.sample_point_num)
        bg_point_coords, bg_point_labels = init_point_sampling(1 - mask_all_image, self.sample_point_num)
        # print(bg_point_coords, bg_point_labels)
        mask = mask > 0
        bbox = batched_mask_to_box(mask.clone().detach().unsqueeze(0))
        bbox = self.sam_trans.apply_boxes_torch(bbox, original_size).squeeze(0)
        point_coords = self.sam_trans.apply_coords_torch(point_coords, original_size)
        bg_point_coords = self.sam_trans.apply_coords_torch(bg_point_coords, original_size)


        # visualization code
        # [bbox_x1, bbox_y1, bbox_x2, bbox_y2] = bbox[0]
        # coor = [(int(bbox_x1), int(bbox_y1)), (int(bbox_x2), int(bbox_y2))]
        
        # to_draw = todo_img.permute(1, 2, 0).type(torch.uint8).numpy()
        # to_draw = cv2.cvtColor(to_draw, cv2.COLOR_RGB2BGR)
        # cv2.circle(to_draw, (int(point_coords[0][0]),int(point_coords[0][1])), 2, (0, 0, 255), 4)
        # cv2.rectangle(to_draw, coor[0], coor[1], (0, 0, 255), 2)

        # cv2.imwrite('img_with_point_prompt.jpg', to_draw)

        return {
            "images": todo_img,
            "label": ann['category_id'],
            "bboxes": bbox,
            # "mask": mask,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "org_sizes": torch.Tensor(original_size),
            "before_pad_size": torch.Tensor(before_pad_size),
            "bg_point_coords": torch.Tensor(bg_point_coords),
        }

    def __len__(self):
        return len(self.coco_pkl)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.target_size - h
        padw = self.target_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


