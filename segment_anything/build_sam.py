# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import importlib
from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, MaskDecoderV0, MaskDecoderV2


def build_sam_vit_h(checkpoint=None, config=None):
    return _build_sam(
        checkpoint=checkpoint,
        config=config,
    )

build_sam = build_sam_vit_h

sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
}

def _build_sam(
    checkpoint=None,
    config=None,
):  
    decoder_classes = {
        "MaskDecoder": MaskDecoder,
        "MaskDecoderV0": MaskDecoderV0,
        "MaskDecoderV2": MaskDecoderV2,
    }
    mask_decoder_class = decoder_classes.get(config.model.mask_decoder.type, None)

    prompt_embed_dim = config.model.image_encoder.out_chans
    image_size = config.model.image_encoder.img_size
    vit_patch_size = config.model.image_encoder.patch_size
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=config.model.image_encoder.depth,
            embed_dim=config.model.image_encoder.embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=config.model.image_encoder.num_heads,
            patch_size=vit_patch_size,
            qkv_bias=config.model.image_encoder.qkv_bias,
            use_rel_pos=config.model.image_encoder.use_rel_pos,
            global_attn_indexes=config.model.image_encoder.global_attn_indexes,
            window_size=config.model.image_encoder.window_size,
            out_chans=prompt_embed_dim,
            cfg=config,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(config.model.pad, config.model.pad),
            mask_in_chans=16,
        ),
        mask_decoder=mask_decoder_class(
            cfg=config,
            num_multimask_outputs=config.model.mask_decoder.num_multimask_outputs,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
                inject_layer=1,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        # import ipdb;ipdb.set_trace()
        new_model_dict = sam.state_dict()
        for k,v in new_model_dict.items():
            if k in state_dict.keys():
                new_model_dict[k] = state_dict[k]
        sam.load_state_dict(new_model_dict)
    if config.tune_with_prompt and config.zero_conv:
        sam.mask_decoder.initzeroconv()
    return sam
