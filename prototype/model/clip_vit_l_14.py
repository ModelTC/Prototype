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


def clip_vit_l_14(
    pretrained="openai", num_classes=1000, checkpoint_path=None, **kwargs
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
        **kwargs: Additional arguments passed to open_clip.create_model_and_transforms
    """
    try:
        import open_clip
    except ImportError:
        raise ImportError(
            "open_clip is required. Install it with: pip install open-clip-torch"
        )

    # Create CLIP model
    if pretrained == "fare2-clip":
        # Load FARE2 model from HuggingFace
        model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:chs20/fare2-clip", device="cpu"
        )
        vision_model = model.visual
    elif pretrained == "tecoa2-clip":
        # Load TeCoA2 model from HuggingFace
        # Note: If HuggingFace model doesn't exist, you can load from checkpoint using pretrain.path in config
        try:
            model, _, _ = open_clip.create_model_and_transforms(
                "hf-hub:chs20/tecoa2-clip", device="cpu"
            )
            vision_model = model.visual
        except Exception:
            # Fallback: create OpenAI model, checkpoint will be loaded via pretrain.path in config
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai", device="cpu"
            )
            vision_model = model.visual
    else:
        # Create standard OpenAI CLIP model
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device="cpu"
        )
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
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = (
                    key.replace("module.", "") if key.startswith("module.") else key
                )
                new_state_dict[new_key] = value

            # Load into vision model
            vision_model.load_state_dict(new_state_dict, strict=False)

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
