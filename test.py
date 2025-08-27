import requests
import base64
from PIL import Image
import io
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def draw_2dbox(img, bboxes, labels=None, output="output_det2d.jpg"):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        top_left = (x1, y1)
        bottom_right = (x2, y2)
        color = (0, 0, 255)  # 红色 (B, G, R)
        thickness = 2
        cv2.rectangle(img_bgr, top_left, bottom_right, color, thickness)
    cv2.imwrite(output, img_bgr)

def draw_seg(img, masks, output="output_seg.jpg"):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for mask in masks:
        mask = np.array(mask, dtype=np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(img_bgr, [contour], -1, (0, 255, 0), 3)  # 绿色轮廓
    cv2.imwrite(output, img_bgr)

location_3d = "http://localhost:8000/location_3d"
location_2d = "http://localhost:8000/location_2d"
location_seg = "http://localhost:8000/location_seg"

img = cv2.imread('images/human.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img.shape[0], img.shape[1]

# 加载图像并转 base64
with open('images/human.jpg', 'rb') as f:
    image_bytes = f.read()
image_b64 = base64.b64encode(image_bytes).decode('utf-8')

data = {
    'image': image_b64,
    'text': "human"
}

res = requests.post(location_seg, json=data)
# draw_2dbox(img, res.json()["bboxes_2d"], "human", output="./images/output_2d.jpg")
result = res.json()
if "error" in result.keys():
    print(result["error"])
else:
    masks = res.json()['masks']
    draw_seg(img, masks, output="./images/output_seg.jpg")