# DetAny3D (Private)

This is a private repository for the **DetAny3D** project, a promptable 3D object detector that builds on SAM, UniDepth, and GroundingDINO.

---

## 🚀 Getting Started

### Step 1: Create Environment

```
conda create -n detany3d python=3.8
conda activate detany3d
```

---

### Step 2: Install Dependencies

#### ✅ (1) Install [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)

Follow the official instructions to install SAM and download its checkpoints.

#### ✅ (2) Install [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)

Follow the UniDepth setup guide to compile and install all necessary packages.

#### ✅ (3) Clone and configure [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

```
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```

> 📌 The exact dependency versions are listed in our `requirements.txt`

---

## 📦 Checkpoints

Please download checkpoints from the following sources and place them under the structure below:

- **DetAny3D / UniDepth / DINO checkpoints**: [Google Drive (link)](https://drive.google.com/drive/folders/17AOq5i1pCTxYzyqb1zbVevPy5jAXdNho?usp=drive_link)
- **SAM checkpoint**: Please download `sam_vit_h.pth` from the official [SAM GitHub Releases](https://github.com/facebookresearch/segment-anything)

```
detany3d_private/
├── checkpoints/
│   ├── sam_ckpts/
│   │   └── sam_vit_h.pth
│   ├── unidepth_ckpts/
│   │   └── unidepth_latest.ckpt
│   ├── dino_ckpts/
│   │   └── dino_swin_large.pth
│   └── detany3d_ckpts/
│       └── detany3d_epoch10.ckpt
```

> GroundingDINO's checkpoint should be downloaded from its [official repo](https://github.com/IDEA-Research/GroundingDINO) and placed as instructed in their documentation.

> All checkpoint paths can be configured inside `./detect_anything/configs/*`.


---

## 📁 Dataset Preparation

**TODO:** Please refer to the dataset preparation instructions (coming soon).

---

## 🏋️‍♂️ Training

```
torchrun --nproc_per_node=1 ./train.py --config_path ./detect_anything/configs/train.yaml
```

---

## 🔍 Inference

```
torchrun --nproc_per_node=1 ./train.py --config_path ./detect_anything/configs/inference.yaml
```

---

## 🌐 Launch Online Demo

```
python ./deploy.py
```

---

For any issues or missing steps, please contact the maintainer or submit a pull request.
