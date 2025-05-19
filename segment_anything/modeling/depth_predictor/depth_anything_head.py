import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from .depth_anything_helpers import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


@MODELS.register_module()
class DPTHead(BaseModule):  # 4 input features with downsample stride as [4, 8, 16, 32]
    def __init__(self,
                 start_level,
                 end_level,
                 in_channels,
                 features=256,
                 use_bn=False,

                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.start_level = start_level
        self.end_level = end_level
        self.in_channels = in_channels
        assert len(in_channels) == (self.end_level - self.start_level)

        self.scratch = _make_scratch(
            in_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, out_features):
        out = []
        for i in range(self.start_level, self.end_level):
            x = out_features[i]
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output(path_1)
        return out, path_1
