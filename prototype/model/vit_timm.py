"""
Vision Transformer models using timm (for compatibility with revisiting-at).
Includes standard ViT and ConvStem (CvSt) variants.
"""

import torch
import torch.nn as nn


def vit_small(drop_path_rate=0.0, **kwargs):
    """ViT-Small using timm"""
    from timm.models import create_model

    model = create_model(
        "vit_small_patch16_224",
        pretrained=False,
        drop_path_rate=drop_path_rate,
        **kwargs
    )
    return model


def vit_base(drop_path_rate=0.0, **kwargs):
    """ViT-Base using timm"""
    from timm.models.vision_transformer import vit_base_patch16_224

    model = vit_base_patch16_224(
        pretrained=False, drop_path_rate=drop_path_rate, **kwargs
    )
    return model


def vit_small_cvst(drop_path_rate=0.0, **kwargs):
    """ViT-Small with ConvStem"""
    from timm.models import create_model
    from .convnext import ConvBlock

    model = create_model(
        "vit_small_patch16_224",
        pretrained=False,
        drop_path_rate=drop_path_rate,
        **kwargs
    )
    model.patch_embed.proj = ConvBlock(48, end_siz=8)
    return model


def vit_base_cvst(drop_path_rate=0.0, **kwargs):
    """ViT-Base with ConvStem"""
    from timm.models.vision_transformer import vit_base_patch16_224
    from .convnext import ConvBlock

    model = vit_base_patch16_224(
        pretrained=False, drop_path_rate=drop_path_rate, **kwargs
    )
    # fin_dim=None will use planes*end_siz = 48*16 = 768 for ViT-Base
    model.patch_embed.proj = ConvBlock(48, end_siz=16, fin_dim=None)
    return model
