import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class AnyDet3DDepthPredictor(BaseModule):

    def __init__(self,
                 src_feature_lvl_idx,  # The index of output feature.
                 depth_head_cfg,

                 depth_encoder_cfg=None,

                 # The max distance of depth embedding,
                 #  typically, for autonomous driving, the maximum depth
                 #  does not surpass 300m. So we set 300m by default.
                 depth_max=300,

                 init_cfg=None):
        """
        Initialize depth predictor and depth encoder
        Args:
            model_cfg [EasyDict]: Depth classification network config
        """
        super().__init__(init_cfg=init_cfg)
        self.src_feature_lvl_idx = src_feature_lvl_idx

        self.depth_encoder_cfg = depth_encoder_cfg
        if self.depth_encoder_cfg is not None:
            self.depth_encoder = MODELS.build(depth_encoder_cfg)

        self.depth_head = MODELS.build(depth_head_cfg)

        self.depth_max = depth_max
        self.depth_pos_embed = nn.Embedding(int(self.depth_max) + 1, 256)

    def forward(self, feature, pos, image_shape):
        depth_logits, depth_embed = self.depth_head(feature,)
        depth_logits = torch.clamp(depth_logits, min=0, max=np.log(self.depth_max))
        depth_pred = depth_logits.exp()  # use exponential to activated the depth value.
        depth_pred = depth_pred.squeeze(1)  # B, H, W

        # depth embeddings with depth positional encodings
        depth_pos_embed_ip = self.interpolate_depth_embed(depth_pred)
        depth_embed = depth_embed + depth_pos_embed_ip

        # Resize the depth_embed to target size.
        target_size = pos[self.src_feature_lvl_idx].shape[2:]
        depth_embed = nn.functional.interpolate(
            depth_embed,
            target_size,
            mode='bilinear',
            align_corners=True)

        if self.depth_encoder_cfg is not None:
            depth_embed_shape = depth_embed.shape
            depth_pos = pos[self.src_feature_lvl_idx].flatten(2)
            depth_pos = depth_pos.repeat(depth_embed.shape[0], 1, 1).permute(0, 2, 1)
            depth_embed = depth_embed.flatten(2).permute(0, 2, 1).contiguous()
            depth_embed = self.depth_encoder(depth_embed, depth_pos)
            depth_embed = depth_embed.permute(0, 2, 1).reshape(depth_embed_shape)
        else:
            depth_embed = depth_embed

        depth_dict = dict(
            embeds=depth_embed,
            depth_pred=depth_pred,
        )
        return depth_dict

    def interpolate_depth_embed(self, depth):
        depth = depth.clamp(min=0, max=self.depth_max)
        pos = self.interpolate_1d(depth, self.depth_pos_embed)
        pos = pos.permute(0, 3, 1, 2)
        return pos

    def interpolate_1d(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=embed.num_embeddings - 1)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta

    def loss(self,
             depth_dict,
             gt_depth_map_list,
             gt_depth_map_valid_flag_list,
             image_shape):

        depth_pred = depth_dict['depth_pred']  # B, H, W

        # 1. Pad gt_depth_map to image_shape
        gt_depth_map = depth_pred.new_zeros(depth_pred.shape[0], *image_shape[:2])
        gt_depth_map_valid_flag = depth_pred.new_zeros(depth_pred.shape[0], *image_shape[:2])
        for i in range(len(gt_depth_map_list)):
            cur_h, cur_w = gt_depth_map_list[i].shape
            gt_depth_map[i, :cur_h, :cur_w] = gt_depth_map_list[i]
            gt_depth_map_valid_flag[i, :cur_h, :cur_w] = gt_depth_map_valid_flag_list[i]

        # 2. Upsample the predicted depth map to ground-truth size.
        depth_pred = nn.functional.interpolate(
            depth_pred.unsqueeze(1),
            gt_depth_map.shape[-2:],
            mode='bilinear',
            align_corners=True).squeeze(1)

        # 3. Collect valid ground-truth & Compute SILog loss.
        depth_pred = depth_pred[gt_depth_map_valid_flag.bool()]
        gt_depth_map = gt_depth_map[gt_depth_map_valid_flag.bool()]
        # SIlog loss.
        alpha, beta = 1e-7, 0.15
        g = torch.log(depth_pred + alpha) - torch.log(gt_depth_map + alpha)
        Dg = torch.var(g) + beta * torch.pow(torch.mean(g), 2)
        loss = 10 * torch.sqrt(Dg)

        return dict(depth_loss=loss)
