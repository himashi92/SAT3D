# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT3D, MaskDecoder3D, PromptEncoder3D, Sam3D, SwinTransformer


def build_sam3D_swin2(checkpoint=None):
    return _build_sam3D_swinc(
        prompt_embed_dim=384,
        encoder_embed_dim=48,
        checkpoint=checkpoint,
    )

build_sam3D = build_sam3D_swin


sam_model_registry3D = {
    "default": build_sam3D_swin,
    "swin_c": build_sam3D_swin,
}


def _build_sam3D_swinc(
    prompt_embed_dim,
    encoder_embed_dim,
    checkpoint=None,
):
    prompt_embed_dim = prompt_embed_dim
    encoder_embed_dim = encoder_embed_dim
    image_size = 128
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam3D(
        image_encoder=SwinTransformer(
            img_size=(128, 128, 128),
            patch_size=(2, 2, 2),
            in_chans=1,
            num_classes=1,
            embed_dim=encoder_embed_dim,
            depths=[2, 2, 2, 1],
            depths_decoder=[1, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=(8, 8, 8),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1,
            final_upsample="expand_first"
        ),
        prompt_encoder=PromptEncoder3D(
            in_chans=1,
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder3D(
            num_multimask_outputs=3,
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict['model_state_dict'])
    return sam
