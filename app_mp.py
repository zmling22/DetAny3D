# process_b.py
from utils import SharedMemoryManager
import numpy as np
import cv2
import sys
import time
from train_utils import *
from wrap_model import WrapModel

from flask import Flask, request, jsonify
from PIL import Image
import cv2
import yaml
import os
import torch.nn as nn
import torch.distributed as dist
from box import Box
import random
import flask
import base64
import multiprocessing as mp
import torch
from groundingdino.util.inference import load_model
from groundingdino.util.inference import predict as dino_predict

from torchvision.ops import box_convert
import colorsys
import json
import hashlib
import io
from segment_anything import SamPredictor, sam_model_registry

# 设置spawn启动方法
mp.set_start_method('spawn', force=True)

def init_models(gpu_id, cfg):
    """初始化模型函数，在每个子进程中调用"""
    print(f"[B-{gpu_id}] 初始化模型...")
    
    # 设置当前进程使用的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(gpu_id)
    
    # 禁用分布式初始化
    torch.distributed.is_available = lambda: False
    torch.distributed.is_initialized = lambda: False
    torch.distributed.get_world_size = lambda group=None: 1
    torch.distributed.get_rank = lambda group=None: 0

    # 加载SAM模型
    my_sam_model = WrapModel(cfg)
    checkpoint = torch.load(cfg.resume, map_location=f'cuda:{gpu_id}')
    new_model_dict = my_sam_model.state_dict()
    for k, v in new_model_dict.items():
        if k in checkpoint['state_dict'].keys() and checkpoint['state_dict'][k].size() == new_model_dict[k].size():
            new_model_dict[k] = checkpoint['state_dict'][k].detach()
    my_sam_model.load_state_dict(new_model_dict)
    my_sam_model.to(f'cuda:{gpu_id}')
    my_sam_model.setup()
    my_sam_model.eval()
    
    # 加载DINO模型
    dino_model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", 
                           "GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")
    dino_model.to(f'cuda:{gpu_id}')
    dino_model.eval()
    
    # 加载SAM predictor
    sam = sam_model_registry["vit_h"](checkpoint=cfg.model.checkpoint)
    sam.to(f'cuda:{gpu_id}')
    predictor = SamPredictor(sam)
    
    sam_trans = ResizeLongestSide(cfg.model.pad)
    
    BOX_TRESHOLD = 0.37
    TEXT_TRESHOLD = 0.25
    
    return my_sam_model, dino_model, predictor, sam_trans, BOX_TRESHOLD, TEXT_TRESHOLD

def worker_process(gpu_id):
    """工作进程函数，每个GPU上运行一个"""
    print(f"[B-{gpu_id}] 启动工作进程，使用GPU: {gpu_id}")
    
    # 在每个子进程中加载配置
    with open('./detect_anything/configs/demo.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg = Box(cfg)
    
    # 初始化模型
    my_sam_model, dino_model, predictor, sam_trans, BOX_TRESHOLD, TEXT_TRESHOLD = init_models(gpu_id, cfg)
    
    import groundingdino.datasets.transforms as T

    def convert_image(img):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_source = Image.fromarray(img, 'RGB')
        image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)
        return image, image_transformed

    def crop_hw(img):
        if img.dim() == 4:
            img = img.squeeze(0)
        h, w = img.shape[1:3]
        assert max(h, w) % 112 == 0, "target_size must be divisible by 112"

        new_h = (h // 14) * 14
        new_w = (w // 14) * 14

        center_h, center_w = h // 2, w // 2

        start_h = center_h - new_h // 2
        start_w = center_w - new_w // 2

        img_cropped = img[:, start_h:start_h + new_h, start_w:start_w + new_w]
        
        return img_cropped.unsqueeze(0)

    def preprocess(x, cfg):
        """Normalize pixel values and pad to a square input."""
        sam_pixel_mean = torch.Tensor(cfg.dataset.pixel_mean).view(-1, 1, 1)
        sam_pixel_std = torch.Tensor(cfg.dataset.pixel_std).view(-1, 1, 1)
        x = (x - sam_pixel_mean) / sam_pixel_std

        h, w = x.shape[-2:]
        padh = cfg.model.pad - h
        padw = cfg.model.pad - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def preprocess_dino(x):
        """Normalize pixel values and pad to a square input."""
        x = x / 255
        IMAGENET_DATASET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        IMAGENET_DATASET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        x = (x - IMAGENET_DATASET_MEAN) / IMAGENET_DATASET_STD
        return x

    def adjust_brightness(color, factor=1.5, v_min=0.3):
        """在 HSV 空间调整亮度，避免过暗"""
        r, g, b = color
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        v = max(v, v_min) * factor
        v = min(v, 1.0)
        return colorsys.hsv_to_rgb(h, s, v)

    def generate_image_token(image: Image.Image) -> str:
        """根据 PIL.Image 生成唯一 Token（SHA-256 哈希值）"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        return hashlib.sha256(img_bytes.getvalue()).hexdigest()

    def predict_2d(input_dict, text):
        with torch.no_grad():
            img, points = input_dict['image'], input_dict['points']

            if img is None:
                raise Exception("No image received")
            
            label_list = []
            bbox_2d_list = []
            
            image_source_dino, image_dino = convert_image(img)
            boxes, logits, phrases = dino_predict(
                model=dino_model,
                image=image_dino,
                caption=text,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD,
                remove_combined=False,
            )

            h, w, _ = image_source_dino.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
            for i, box in enumerate(xyxy):
                bbox_2d_list.append(box.to(torch.int).cpu().numpy().tolist())
                label_list.append(phrases[i])
            
        return bbox_2d_list, label_list

    def predict_3d(input_dict, text):
        with torch.no_grad():
            img, points = input_dict['image'], input_dict['points']

            if img is None:
                raise Exception("No image received")
            
            label_list = []
            bbox_2d_list = []
            point_coords_list = []
            for point in points:
                if point[2] == 1 or point[5] == 1:
                    human_prompt_coord = np.array([point[0], point[1]])
                    point_coords_tensor = torch.tensor(human_prompt_coord, dtype=torch.int).unsqueeze(0)
                    point_coords_list.append(point_coords_tensor)
                    
                if point[2] == 2 and point[5] == 3:
                    x1, y1 = point[:2]
                    x2, y2 = point[3:5]
                    bbox_2d = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                    bbox_2d_list.append(bbox_2d)
                    label_list.append("Unknow")
            
            if len(bbox_2d_list) > 0 and len(point_coords_list) > 0:
                raise Exception("Can not hadle bounding box and point at the same time.")
            
            if len(point_coords_list) == 0:
                mode = 'box'
            else:
                mode = 'point'
                label_list = ["Unknow"]

            image_source_dino, image_dino = convert_image(img)
            if text != '' and mode == 'point':
                print("Both text and point prompt input, follow the point prompt")
                
            boxes, logits, phrases = dino_predict(
                model=dino_model,
                image=image_dino,
                caption=text,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD,
                remove_combined=False,
            )

            h, w, _ = image_source_dino.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
            for i, box in enumerate(xyxy):
                if mode == 'box':
                    bbox_2d_list.append(box.to(torch.int).cpu().numpy().tolist())
                    label_list.append(phrases[i])
                elif mode == 'point':
                    pass
            
            if len(bbox_2d_list) == 0 and len(point_coords_list) == 0:
                raise Exception("No objects found in the image. Please try again with a different prompt.")
                
            raw_img = img.copy()
            original_size = tuple(img.shape[:-1])
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
            img = img.unsqueeze(0)

            img = sam_trans.apply_image_torch(img)
            img = crop_hw(img)
            before_pad_size = tuple(img.shape[2:])
            
            img_for_sam = preprocess(img, cfg).to(f'cuda:{gpu_id}')
            img_for_dino = preprocess_dino(img).to(f'cuda:{gpu_id}')

            resize_ratio = max(img_for_sam.shape) / max(original_size)
            image_h, image_w = int(before_pad_size[0]), int(before_pad_size[1])

            if cfg.model.vit_pad_mask:
                vit_pad_size = (before_pad_size[0] // cfg.model.image_encoder.patch_size, before_pad_size[1] // cfg.model.image_encoder.patch_size)
            else:
                vit_pad_size = (cfg.model.pad // cfg.model.image_encoder.patch_size, cfg.model.pad // cfg.model.image_encoder.patch_size)

            if mode == 'box':
                bbox_2d_tensor = torch.tensor(bbox_2d_list)
                bbox_2d_tensor = sam_trans.apply_boxes_torch(bbox_2d_tensor, original_size).to(torch.int).to(f'cuda:{gpu_id}')
                input_dict = {
                    "images": img_for_sam,
                    "vit_pad_size": torch.tensor(vit_pad_size).to(f'cuda:{gpu_id}').unsqueeze(0),
                    "images_shape": torch.Tensor(before_pad_size).to(f'cuda:{gpu_id}').unsqueeze(0),
                    "image_for_dino": img_for_dino,
                    "boxes_coords": bbox_2d_tensor,
                }
            if mode == 'point':
                points_2d_tensor = torch.stack(point_coords_list, dim=1).to(f'cuda:{gpu_id}')
                points_2d_tensor = sam_trans.apply_coords_torch(points_2d_tensor, original_size)
                input_dict = {
                    "images": img_for_sam,
                    'vit_pad_size': torch.tensor(vit_pad_size).to(f'cuda:{gpu_id}').unsqueeze(0),
                    "images_shape": torch.Tensor(before_pad_size).to(f'cuda:{gpu_id}').unsqueeze(0),
                    "image_for_dino": img_for_dino,
                    "point_coords": points_2d_tensor,
                }

            ret_dict = my_sam_model(input_dict)

            K = ret_dict['pred_K']
            decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes(ret_dict, cfg, K)
            rot_mat = rotation_6d_to_matrix(ret_dict['pred_pose_6d'])

        return decoded_bboxes_pred_3d, rot_mat

    def predict_seg(input_dict, text):
        with torch.no_grad():
            img, points = input_dict['image'], input_dict['points']

            if img is None:
                raise Exception("No image received")
            
            label_list = []
            bbox_2d_list = []

            image_source_dino, image_dino = convert_image(img)
            boxes, logits, phrases = dino_predict(
                model=dino_model,
                image=image_dino,
                caption=text,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD,
                remove_combined=False,
            )

            h, w, _ = image_source_dino.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
            for i, box in enumerate(xyxy):
                bbox_2d_list.append(box.to(torch.int).cpu().numpy().tolist())
                label_list.append(phrases[i])
            
            if len(bbox_2d_list) == 0:
                raise Exception("No objects found in the image. Please try again with a different prompt.")
            
            predictor.set_image(image_source_dino)
            masks_result = []

            for bbox in bbox_2d_list:
                masks, _, _ = predictor.predict(box=np.array(bbox))
                mask = np.zeros_like(masks[0])
                for i in range(masks.shape[0]):
                    mask = mask | masks[i]
                masks_result.append((mask.astype(np.uint8) * 255).tolist())

        return masks_result

    # 配置参数
    SHM_SIZE = 1024 * 1024 * 512

    try:
        # 为每个GPU创建独立的共享内存
        print(f"[B-{gpu_id}] 初始化共享内存和信号量...")
        shm_img = SharedMemoryManager(f"image_data_{gpu_id}", SHM_SIZE, create=True, is_server=True)
        shm_result = SharedMemoryManager(f"result_data_{gpu_id}", SHM_SIZE, create=True, is_server=True)

        while True:
            try:
                # 等待客户端数据
                print(f"\n[B-{gpu_id}] 等待客户端数据...")
                data = shm_img.read_data()
                
                endpoint = data.get('endpoint', None)
                
                # 处理数据
                print(f"[B-{gpu_id}] 处理图像中...")
                input_dict = {
                    'image': data['image'],
                    'points': data.get('points', [])
                }
                text = data.get('text', '')

                if endpoint == "location_3d":
                    decoded_bboxes_pred_3d, rot_mat = predict_3d(input_dict, text)
                    result = {
                        'bboxes_3d': decoded_bboxes_pred_3d,
                        'rot_mat': rot_mat.tolist(),
                        'text': text
                    }
                elif endpoint == "location_2d":
                    bbox_2d_list, label_list = predict_2d(input_dict, text)
                    result = {
                        'bboxes_2d': bbox_2d_list,
                        'labels': label_list,
                        'text': text
                    }
                elif endpoint == "location_seg":
                    masks_result = predict_seg(input_dict, text)
                    result = {
                        'masks': masks_result,
                        'text': text
                    }
                else:
                    result = {"error": f"Unknown endpoint: {endpoint}"}
                
                # 返回结果
                print(f"[B-{gpu_id}] 返回处理结果")
                shm_result.write_data(result)
                shm_result.notify_done()

            except KeyboardInterrupt:
                print(f"\n[B-{gpu_id}] 收到终止信号，退出循环")
                break
            except Exception as e:
                print(f"[B-{gpu_id}] 错误: {str(e)}")
                import traceback
                traceback.print_exc()
                shm_result.write_data({"error": str(e)})
                shm_result.notify_done()
                time.sleep(1)

    finally:
        print(f"[B-{gpu_id}] 工作进程退出")

if __name__ == "__main__":
    # 获取可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 张GPU")
    
    if num_gpus == 0:
        print("未检测到GPU，无法启动工作进程")
        sys.exit(1)
    
    # 创建进程列表
    processes = []
    
    num_gpus = 4

    # 为每个GPU启动一个工作进程
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_process, args=(gpu_id,))
        p.daemon = False  # 设置为非守护进程，确保正常退出
        processes.append(p)
        print(f"启动GPU {gpu_id} 的工作进程")
    
    # 启动所有进程
    for p in processes:
        p.start()
    
    # 等待所有进程结束
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n主进程收到终止信号，终止所有工作进程")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)
    
    print("主进程退出")
