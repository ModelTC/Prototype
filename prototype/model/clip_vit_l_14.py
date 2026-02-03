"""
CLIP ViT-L/14 model for RobustART.
Supports loading different pretrained weights:
- fare2-clip: FARE2 model from HuggingFace
- tecoa2-clip: TeCoA2 model from HuggingFace
- ViT-L-14.pt: Standard OpenAI CLIP model checkpoint
"""

import torch
import torch.nn as nn
import os
import inspect


def clip_vit_l_14(
    pretrained="openai",
    num_classes=1000,
    checkpoint_path=None,
    use_pretrain_path=False,
    **kwargs,
):
    """
    CLIP ViT-L/14 model.

    Args:
        pretrained (str): Pretrained model type. Options:
            - 'openai': Standard OpenAI CLIP ViT-L/14
            - 'fare2-clip': FARE2 robust CLIP from HuggingFace (hf-hub:chs20/fare2-clip)
            - 'tecoa2-clip': TeCoA2 robust CLIP from HuggingFace (hf-hub:chs20/tecoa2-clip)
            - 'openai': Standard OpenAI CLIP (default)
        num_classes (int): Number of classes for classification head. Default: 1000
        checkpoint_path (str): Optional path to checkpoint file. If provided and file exists,
            will load visual encoder weights from checkpoint. This is useful for loading
            fare_eps_2.pt, tecoa_eps_2.pt, etc. that only contain visual encoder weights.
        use_pretrain_path (bool): If True, use saver.pretrain.path from config to load checkpoint.
            Default: False (automatically download from HuggingFace).
        **kwargs: Additional arguments passed to open_clip.create_model_and_transforms
    """
    # Check if we should use pretrain path from config
    if use_pretrain_path:
        full_config = kwargs.pop("_full_config", None)
        if full_config is not None:
            try:
                # Try to access saver.pretrain.path from full_config
                if (
                    hasattr(full_config, "saver")
                    and hasattr(full_config.saver, "pretrain")
                    and hasattr(full_config.saver.pretrain, "path")
                ):
                    pretrain_path = full_config.saver.pretrain.path
                    if pretrain_path and os.path.exists(pretrain_path):
                        checkpoint_path = pretrain_path
            except (AttributeError, KeyError):
                pass

    try:
        import open_clip
    except ImportError:
        raise ImportError(
            "open_clip is required. Install it with: pip install open-clip-torch"
        )

    def _create_open_clip(model_name, pretrained_name=None):
        kwargs = {"device": "cpu"}
        if pretrained_name is not None:
            kwargs["pretrained"] = pretrained_name
        # PyTorch 2.6 defaults weights_only=True; disable if supported
        sig = inspect.signature(open_clip.create_model_and_transforms)
        if "weights_only" in sig.parameters:
            kwargs["weights_only"] = False
        return open_clip.create_model_and_transforms(model_name, **kwargs)

    # Create CLIP model
    # If use_pretrain_path is True, always create base OpenAI model and load from checkpoint
    if use_pretrain_path and checkpoint_path and os.path.exists(checkpoint_path):
        model, _, _ = _create_open_clip("ViT-L-14", "openai")
        vision_model = model.visual
    elif pretrained == "fare2-clip":
        # Load FARE2 model from HuggingFace
        model, _, _ = _create_open_clip("hf-hub:chs20/fare2-clip")
        vision_model = model.visual
    elif pretrained == "tecoa2-clip":
        # Load TeCoA2 model from HuggingFace
        model, _, _ = _create_open_clip("hf-hub:chs20/tecoa2-clip")
        vision_model = model.visual
    else:
        # Create standard OpenAI CLIP model
        model, _, _ = _create_open_clip("ViT-L-14", "openai")
        vision_model = model.visual

    # If checkpoint_path is provided and exists, load visual encoder weights
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # If checkpoint has 'model' key, use that
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            # If checkpoint has 'visual' key, use that
            elif "visual" in checkpoint:
                state_dict = checkpoint["visual"]
            # Otherwise, assume the checkpoint is the state dict itself
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "") if key.startswith("module.") else key
            cleaned_state_dict[new_key] = value

        # Filter to vision-only keys when loading a full CLIP checkpoint
        vision_sd = vision_model.state_dict()
        vision_keys = set(vision_sd.keys())
        vision_state = {}
        skipped = []
        for key, value in cleaned_state_dict.items():
            if key.startswith("visual."):
                vkey = key[len("visual.") :]
            else:
                vkey = key
            if vkey in vision_keys:
                # Only keep matching shapes to avoid size mismatch
                if vision_sd[vkey].shape == value.shape:
                    vision_state[vkey] = value
                else:
                    skipped.append((vkey, tuple(value.shape), tuple(vision_sd[vkey].shape)))

        # If we couldn't match any keys, fall back to the cleaned dict
        if vision_state:
            vision_model.load_state_dict(vision_state, strict=False)
            if skipped:
                print(
                    f"[clip_vit_l_14] Skipped {len(skipped)} mismatched keys (e.g. {skipped[0]})"
                )
        else:
            vision_model.load_state_dict(cleaned_state_dict, strict=False)

    # Add classification head if needed
    if num_classes != 0:
        # Get the output dimension of the vision encoder
        # For ViT-L/14, the output dimension is 1024
        embed_dim = (
            vision_model.output_dim if hasattr(vision_model, "output_dim") else 1024
        )

        # Create a wrapper that includes the classification head
        class CLIPVisionClassifier(nn.Module):
            def __init__(self, vision_model, num_classes):
                super().__init__()
                self.vision_model = vision_model
                self.head = nn.Linear(embed_dim, num_classes)
                # Store reference to visual for checkpoint loading compatibility
                self.visual = vision_model

            def forward(self, x):
                x = self.vision_model(x)
                x = self.head(x)
                return x

            def forward_features(self, x):
                """Return features before classification head"""
                return self.vision_model(x)

            def load_state_dict(self, state_dict, strict=True):
                """
                Custom load_state_dict that handles checkpoints containing only visual encoder weights.
                If state_dict keys don't match model structure, try loading into vision_model.
                This handles checkpoints like fare_eps_2.pt, tecoa_eps_2.pt that only contain visual encoder weights.
                """
                # First try normal loading
                missing_keys, unexpected_keys = [], []
                try:
                    missing_keys, unexpected_keys = super().load_state_dict(
                        state_dict, strict=False
                    )
                    # If strict=True and there are missing/unexpected keys, raise error
                    if strict and (missing_keys or unexpected_keys):
                        raise RuntimeError(
                            f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}"
                        )
                    return
                except (RuntimeError, KeyError) as e:
                    # If normal loading fails, check if keys match vision_model structure directly
                    # This handles checkpoints that only contain visual encoder weights (like fare_eps_2.pt)
                    vision_keys = [
                        k
                        for k in state_dict.keys()
                        if not k.startswith("head.")
                        and not k.startswith("vision_model.")
                        and not k.startswith("visual.")
                    ]

                    if vision_keys:
                        # Check if these keys match vision_model structure
                        try:
                            # Try loading directly into vision_model
                            vision_state = {
                                k.replace("module.", ""): v
                                for k, v in state_dict.items()
                                if not k.startswith("head.")
                            }
                            # Remove prefixes if present
                            vision_state_clean = {}
                            for k, v in vision_state.items():
                                new_k = (
                                    k.replace("vision_model.", "")
                                    .replace("visual.", "")
                                    .replace("module.", "")
                                )
                                vision_state_clean[new_k] = v

                            self.vision_model.load_state_dict(
                                vision_state_clean, strict=False
                            )

                            # If there are head weights, load them
                            head_state = {
                                k.replace("head.", "").replace("module.", ""): v
                                for k, v in state_dict.items()
                                if k.startswith("head.")
                            }
                            if head_state:
                                self.head.load_state_dict(head_state, strict=False)

                            return
                        except Exception:
                            pass

                    # If all else fails and strict=True, raise the original error
                    if strict:
                        raise e
                    # Otherwise, just return (non-strict loading)
                    return

        model = CLIPVisionClassifier(vision_model, num_classes)
    else:
        # Return vision model directly without classification head
        model = vision_model

    return model
