# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d
import copy
from .mask_decoder import MaskDecoder


class MaskDecoderV0(nn.Module):
    def __init__(
        self,
        *,
        cfg,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()

        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # bbox tokens
        self.box_token = nn.Embedding(1, transformer_dim)
        self.box_token_3d = nn.Embedding(1, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        # sam original head
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        
        # bbox related heads
        self.bbox_head = MLP(
            transformer_dim, iou_head_hidden_dim, 6, iou_head_depth + 1
        )
        self.bbox_3d_center_head = MLP(
            transformer_dim, iou_head_hidden_dim, 2, iou_head_depth
        )
        self.bbox_3d_depth_head = MLP(
            transformer_dim, iou_head_hidden_dim, 2, iou_head_depth
        )
        self.bbox_3d_dims_head = MLP(
            transformer_dim, iou_head_hidden_dim, 3, iou_head_depth
        )
        self.bbox_3d_alpha_cls_head = MLP(
            transformer_dim, iou_head_hidden_dim, 24, iou_head_depth
        )
        self.bbox_3d_alpha_res_head = MLP(
            transformer_dim, iou_head_hidden_dim, 24, iou_head_depth
        )

        self.cfg = cfg

        if cfg.merge_dino_feature:
            self.zero_conv2d_cam = nn.Conv2d(2048, 256, 1)
            self.zero_conv2d_metric = nn.Conv2d(1024, 256, 1)
            self.zero_conv2d = nn.Conv2d(512, 256, 1)
        else:
            self.zero_conv2d_cam = nn.Conv2d(2560, 256, 1)
            self.zero_conv2d_metric = nn.Conv2d(1280, 256, 1)
            self.zero_conv2d = nn.Conv2d(640, 256, 1)

        self.out_zero_conv2d = nn.Conv2d(256, 256, 1)

        if cfg.zero_conv:

            for p in self.zero_conv2d_cam.parameters():
                p.detach().zero_()

            for p in self.zero_conv2d.parameters():
                p.detach().zero_()

            for p in self.zero_conv2d_metric.parameters():
                p.detach().zero_()

            for p in self.out_zero_conv2d.parameters():
                p.detach().zero_()

    def initzeroconv(self):
        self.transformer2 = copy.deepcopy(self.transformer)
        self.transformer2.is_copy = True

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        metric_feature,
        camera_feature,
        depth_feature,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        # rays: torch.Tensor,

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        
        masks, iou_pred, pred_bbox_2d, pred_center_2d, pred_bbox_3d_depth, pred_bbox_3d_depth_log_variance, pred_bbox_3d_dims, pred_bbox_3d_alpha_cls, pred_bbox_3d_alpha_res = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            metric_feature=metric_feature,
            camera_feature=camera_feature,
            depth_feature=depth_feature,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            # rays=rays,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred, pred_bbox_2d, pred_center_2d, pred_bbox_3d_depth, pred_bbox_3d_depth_log_variance, pred_bbox_3d_dims, pred_bbox_3d_alpha_cls, pred_bbox_3d_alpha_res

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        metric_feature,
        camera_feature,
        depth_feature,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        # rays: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        
        metric_embeddings = self.zero_conv2d_metric(metric_feature)
        camera_embeddings = self.zero_conv2d_cam(camera_feature)
        depth_embeddings = self.zero_conv2d(depth_feature)

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.box_token.weight, self.box_token_3d.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)

        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if self.cfg.zero_conv:
            control_tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = image_embeddings + dense_prompt_embeddings[..., :image_embeddings.shape[-2], :image_embeddings.shape[-1]]
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape 

        control_q = None
        control_k = None
        
        hs, src = self.transformer(src + depth_embeddings + metric_embeddings + camera_embeddings, pos_src, tokens, control_q, control_k)

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        box_tokens_out = hs[:, 1 + self.num_mask_tokens, :]
        box_3d_tokens_out = hs[:, 2 + self.num_mask_tokens, :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        
        for i in range(self.num_mask_tokens):
            # with torch.no_grad():
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        
        hyper_in = torch.stack(hyper_in_list, dim=1)

        #  2d bbox
        pred_bbox_2d = self.bbox_head(box_tokens_out).sigmoid()
        # 3d bbox
        # pred_center_2d = self.bbox_3d_center_head(box_3d_tokens_out)
        pred_bbox_3d_depth_hs = self.bbox_3d_depth_head(box_3d_tokens_out)
        pred_bbox_3d_dims = self.bbox_3d_dims_head(box_3d_tokens_out)
        pred_bbox_3d_alpha_hs = self.bbox_3d_alpha_cls_head(box_3d_tokens_out)

        pred_center_2d = pred_bbox_2d[..., :2]
        pred_bbox_2d = pred_bbox_2d[..., 2:]

        pred_bbox_3d_depth = pred_bbox_3d_depth_hs[..., :1]
        pred_bbox_3d_depth_log_variance = pred_bbox_3d_depth_hs[..., 1:]
        
        pred_bbox_3d_alpha_cls = pred_bbox_3d_alpha_hs[..., :12]
        pred_bbox_3d_alpha_res = pred_bbox_3d_alpha_hs[..., 12:]

        # output sam masks
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred, pred_bbox_2d, pred_center_2d, pred_bbox_3d_depth, pred_bbox_3d_depth_log_variance, pred_bbox_3d_dims, pred_bbox_3d_alpha_cls, pred_bbox_3d_alpha_res


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = x.sigmoid()
        return x
