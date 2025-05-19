import os, pickle, numpy as np, cv2
from typing import List
from segment_anything.datasets.utils import *

# ---------- 配置区 ----------
IMAGE_DIR = "/cpfs01/user/zhanghanxue/segment-anything/data/kitti/testing/image_2"
CALIB_DIR = "/cpfs01/user/zhanghanxue/segment-anything/data/kitti/testing/calib"
PRED_DIR  = "/cpfs01/user/zhanghanxue/VirConv/output/models/kitti/VirConv-S/default/eval/epoch_2/test/default/final_result/data"  # ← 模型输出的 txt
OUT_PKL   = "ret_list_2.pkl"
# --------------------------------

# --------------------------------

# 把 KITTI 类别映射成整数 label_id，可按需增删
CLASS2ID = {
    "Car": 1, "Van": 1,            # 你只关心 coarse-class 时可合并
    "Pedestrian": 2, "Person": 2,
    "Cyclist": 3,
    "Truck": 4, "Bus": 4,
    "Misc": 0,  "DontCare": 0,
}

def load_P2_from_calib(calib_path: str) -> np.ndarray:
    with open(calib_path) as f:
        for ln in f:
            if ln.startswith("P2:"):
                vals = list(map(float, ln.split()[1:]))
                return np.array(vals, dtype=np.float32).reshape(3, 4)[:, :3]
    return None

def parse_kitti_pred(txt_path: str) -> List[dict]:
    """读取单张图像的 txt，返回 obj_list"""
    obj_list = []
    with open(txt_path) as f:
        for l in f:
            if not l.strip(): continue
            ss = l.strip().split()
            # KITTI 3D 检测格式：type, trunc, occ, alpha, 4×bbox2d, 3×dim, 3×loc, ry, [score]
            cls            = ss[0]
            trunc, occ     = float(ss[1]), int(ss[2])
            bbox2d_tight   = list(map(float, ss[4:8]))          # u_min,v_min,u_max,v_max
            h, w, l3       = map(float, ss[8:11])               # height, width, length
            x, y, z        = map(float, ss[11:14])              # 相机坐标系
            yaw            = float(ss[14])                      # rotation_y
            score          = float(ss[15]) if len(ss) > 15 else 1.0

            obj_list.append({
                "raw":         (cls, trunc, occ, bbox2d_tight, h, w, l3, x, y, z, yaw, score)
            })
    return obj_list

ret_list = []

for fname in sorted(os.listdir(IMAGE_DIR)):
    # import ipdb;ipdb.set_trace()
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_id   = os.path.splitext(fname)[0]
    img_path = os.path.join(IMAGE_DIR, fname)

    # --- 相机内参 ---
    K = None
    txt_calib = os.path.join(CALIB_DIR, img_id + ".txt")
    if os.path.isfile(txt_calib):
        K = load_P2_from_calib(txt_calib)
    K_arr = np.array([K], dtype=np.float32)        # 与原脚本保持一致 (1,3,3)

    # --- 解析预测 ---
    pred_path  = os.path.join(PRED_DIR, img_id + ".txt")
    obj_list   = []
    if os.path.isfile(pred_path):
        pred_objs = parse_kitti_pred(pred_path)
        for ob in pred_objs:
            cls, trunc, occ, bbox2d_tight, h, w, l3, x, y, z, yaw, score = ob["raw"]
            y = y - h/2
            yaw = yaw + np.pi / 2

            # # 3D -> 八点 -> 2D
            # if K is not None:
            #     verts3d, _      = compute_3d_bbox_vertices(x, y, z, w, h, l3, yaw)
            #     verts2d         = project_to_image(verts3d, K)
            #     xmin, ymin      = verts2d.min(axis=0)
            #     xmax, ymax      = verts2d.max(axis=0)
            #     bbox2d_proj     = [float(xmin), float(ymin), float(xmax), float(ymax)]
            # else:
            #     bbox2d_proj     = [-1, -1, -1, -1]
            bbox2d_proj     = [-1, -1, -1, -1]
            obj_list.append({
                "3d_bbox":       [x, y, z, w, h, l3, yaw],
                "2d_bbox_proj":  bbox2d_proj,
                "2d_bbox_tight": bbox2d_tight,
                "2d_bbox_trunc": bbox2d_tight,   # 可保留同值
                "label":         CLASS2ID.get(cls, 0),
                "rotation_pose": None,
                "instance_id":   0,
                "score":         score,
                "image_id":      img_id,
                "visibility":    occ,
                "truncation":    trunc,
            })
            # image = cv2.imread(img_path)
            # vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l3, yaw)
            # K = K_arr
            # vertices_2d = project_to_image(vertices_3d, K.squeeze(0))
            # fore_plane_center_2d = project_to_image(fore_plane_center_3d, K.squeeze(0))
            # # # import ipdb;ipdb.set_trace()
            # draw_bbox_2d(image, vertices_2d)
            # cv2.circle(image, fore_plane_center_2d[0].astype(int), 2, (0, 0, 255) , 1)
            # cv2.imwrite('3D_test.png', image)
            # import ipdb;ipdb.set_trace()

    # 若没有检测结果，保持一个默认空框
    if not obj_list:
        obj_list = [{
            "3d_bbox": [-1]*7, "2d_bbox_proj": [-1]*4, "2d_bbox_tight": [-1]*4,
            "2d_bbox_trunc": [-1]*4, "label": 0, "rotation_pose": None,
            "instance_id": 0, "score": 1.0, "image_id": img_id,
            "visibility": -1, "truncation": -1,
        }]

    # ---- 记录 ----
    ret_list.append({
        "K":        K_arr,
        "obj_list": obj_list,
        "img_path": img_path,
        "depth_path": None,
        "imgIds":   img_id,
    })

# 保存
with open(OUT_PKL, "wb") as f:
    pickle.dump(ret_list, f)

print(f"共处理 {len(ret_list)} 张图，结果已保存到 {OUT_PKL}")
