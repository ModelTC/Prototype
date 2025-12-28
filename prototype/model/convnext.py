"""
ConvNeXt models based on revisiting-at implementation.
Uses timm for standard ConvNeXt models and custom ConvStem variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from prototype.prototype.utils.trunc_normal_initializer import trunc_normal_


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvBlock1(nn.Module):
    """ConvStem for ConvNext-Tiny and ConvNext-Small"""

    def __init__(self, siz=48, end_siz=8, fin_dim=384):
        super(ConvBlock1, self).__init__()
        self.planes = siz
        # fin_dim parameter is not used in ConvBlock1, output is always planes*2
        # But we keep it for compatibility with revisiting-at code
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.planes, kernel_size=3, stride=2, padding=1),
            LayerNorm(self.planes, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(self.planes, self.planes * 2, kernel_size=3, stride=2, padding=1),
            LayerNorm(self.planes * 2, data_format="channels_first"),
            nn.GELU(),
        )

    def forward(self, x):
        return self.stem(x)


class ConvBlock3(nn.Module):
    """ConvStem for ConvNext-Base and ConvNext-Large"""

    def __init__(self, siz=64):
        super(ConvBlock3, self).__init__()
        self.planes = siz
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.planes, kernel_size=3, stride=2, padding=1),
            LayerNorm(self.planes, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(
                self.planes, int(self.planes * 1.5), kernel_size=3, stride=2, padding=1
            ),
            LayerNorm(int(self.planes * 1.5), data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(
                int(self.planes * 1.5),
                self.planes * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            LayerNorm(self.planes * 2, data_format="channels_first"),
            nn.GELU(),
        )

    def forward(self, x):
        return self.stem(x)


class ConvBlock(nn.Module):
    """ConvStem for ViT models"""

    def __init__(self, siz=48, end_siz=8, fin_dim=384):
        super(ConvBlock, self).__init__()
        self.planes = siz
        # Original logic: use planes*end_siz if fin_dim != 432, else use 432
        # Also handle None case (for ViT-Base, fin_dim=None means use planes*end_siz)
        if fin_dim is None or fin_dim != 432:
            fin_dim = self.planes * end_siz
        else:
            fin_dim = 432
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.planes, kernel_size=3, stride=2, padding=1),
            LayerNorm(self.planes, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(self.planes, self.planes * 2, kernel_size=3, stride=2, padding=1),
            LayerNorm(self.planes * 2, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(
                self.planes * 2, self.planes * 4, kernel_size=3, stride=2, padding=1
            ),
            LayerNorm(self.planes * 4, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(
                self.planes * 4, self.planes * 8, kernel_size=3, stride=2, padding=1
            ),
            LayerNorm(self.planes * 8, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(self.planes * 8, fin_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.stem(x)


class ConvBlock2(nn.Module):
    """ConvStem for ViT-Medium"""

    def __init__(self, siz=48, end_siz=8, fin_dim=384):
        super(ConvBlock2, self).__init__()
        self.planes = siz
        fin_dim = self.planes * end_siz if fin_dim != 432 else 432
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.planes, kernel_size=3, stride=2, padding=1),
            LayerNorm(self.planes, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(self.planes, self.planes * 2, kernel_size=3, stride=2, padding=1),
            LayerNorm(self.planes * 2, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(
                self.planes * 2, self.planes * 4, kernel_size=3, stride=2, padding=1
            ),
            LayerNorm(self.planes * 4, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(
                self.planes * 4, self.planes * 8, kernel_size=3, stride=2, padding=1
            ),
            LayerNorm(self.planes * 8, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(self.planes * 8, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.stem(x)


def convnext_tiny(drop_path_rate=0.0, **kwargs):
    """ConvNeXt-Tiny using timm"""
    import timm

    model = timm.models.convnext.convnext_tiny(
        pretrained=False, drop_path_rate=drop_path_rate, **kwargs
    )
    return model


def convnext_small(drop_path_rate=0.0, **kwargs):
    """ConvNeXt-Small using timm"""
    import timm

    model = timm.models.convnext.convnext_small(
        pretrained=False, drop_path_rate=drop_path_rate, **kwargs
    )
    return model


def convnext_base(drop_path_rate=0.0, **kwargs):
    """ConvNeXt-Base using timm"""
    import timm

    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
    model = timm.models.convnext._create_convnext(
        "convnext_base.fb_in1k",
        pretrained=False,
        drop_path_rate=drop_path_rate,
        **model_args,
        **kwargs
    )
    return model


def convnext_large(drop_path_rate=0.0, **kwargs):
    """ConvNeXt-Large using timm"""
    import timm

    model = timm.models.convnext.convnext_large(
        pretrained=False, drop_path_rate=drop_path_rate, **kwargs
    )
    return model


def convnext_tiny_cvst(drop_path_rate=0.0, **kwargs):
    """ConvNeXt-Tiny with ConvStem"""
    import timm

    model = timm.models.convnext.convnext_tiny(
        pretrained=False, drop_path_rate=drop_path_rate, **kwargs
    )
    model.stem = ConvBlock1(48, end_siz=8)
    return model


def convnext_small_cvst(drop_path_rate=0.0, **kwargs):
    """ConvNeXt-Small with ConvStem"""
    import timm

    model = timm.models.convnext.convnext_small(
        pretrained=False, drop_path_rate=drop_path_rate, **kwargs
    )
    model.stem = ConvBlock1(48, end_siz=8)
    return model


def convnext_base_cvst(drop_path_rate=0.0, **kwargs):
    """ConvNeXt-Base with ConvStem"""
    import timm

    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
    model = timm.models.convnext._create_convnext(
        "convnext_base.fb_in1k",
        pretrained=False,
        drop_path_rate=drop_path_rate,
        **model_args,
        **kwargs
    )
    model.stem = ConvBlock3(64)
    return model


def convnext_large_cvst(drop_path_rate=0.0, **kwargs):
    """ConvNeXt-Large with ConvStem"""
    import timm

    model = timm.models.convnext.convnext_large(
        pretrained=False, drop_path_rate=drop_path_rate, **kwargs
    )
    model.stem = ConvBlock3(96)
    return model
