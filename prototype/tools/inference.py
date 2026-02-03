import os
import argparse
import pprint
import torch
import json
import cv2
import numpy as np
import prototype.spring.linklink as link

import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from easydict import EasyDict
from torch.autograd import Variable

from prototype.prototype.solver.cls_solver import ClsSolver
from prototype.prototype.utils.dist import link_dist
from prototype.prototype.utils.misc import (
    makedir,
    create_logger,
    get_logger,
    modify_state,
)
from prototype.prototype.data import build_imagenet_test_dataloader
from prototype.prototype.data import build_custom_dataloader

# set seed

import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class Inference(ClsSolver):
    def __init__(self, config):
        self.image_dir = config["image_dir"]
        # import ipdb; ipdb.set_trace()
        self.meta_file = config.get("meta_file", "")
        if self.meta_file == None:
            self.meta_file = ""

        self.output = config.get("output", "inference_results")
        self.recover = config.get("recover", "")
        self.cam = config.get("cam", False)
        self.attn_rollout = config.get("attn_rollout", False)
        self.visualize = config.get("visualize", False)
        self.sample = config.get("sample", -1)
        self.feature_name = config.get("name", "module.layer4")
        if "module" not in self.feature_name:
            self.feature_name = "module." + self.feature_name
        self.feature = None
        self.gradient = None
        super(Inference, self).__init__(config["config"])

    def setup_env(self):
        # dist
        self.dist = EasyDict()
        self.dist.rank, self.dist.world_size = link.get_rank(), link.get_world_size()
        self.prototype_info.world_size = self.dist.world_size
        # directories
        self.path = EasyDict()
        self.path.root_path = os.path.dirname(self.config_file)
        self.path.result_path = os.path.abspath(self.output)
        makedir(self.path.result_path)
        # logger
        create_logger(os.path.join(self.path.root_path, "log.txt"))
        self.logger = get_logger(__name__)
        self.logger.info(f"config: {pprint.pformat(self.config)}")
        if "SLURM_NODELIST" in os.environ:
            self.logger.info(f"hostnames: {os.environ['SLURM_NODELIST']}")
        # load pretrain checkpoint
        if self.recover != "":
            self.state = torch.load(self.recover, "cpu")
            self.logger.info(
                f"Recovering from {self.recover}, keys={list(self.state.keys())}"
            )

        elif hasattr(self.config.saver, "pretrain"):
            self.state = torch.load(self.config.saver.pretrain.path, "cpu")
            self.logger.info(
                f"Recovering from {self.config.saver.pretrain.path}, keys={list(self.state.keys())}"
            )
            if hasattr(self.config.saver.pretrain, "ignore"):
                self.state = modify_state(self.state, self.config.saver.pretrain.ignore)
        else:
            self.state = {}
            self.state["last_iter"] = 0
        # others
        torch.backends.cudnn.benchmark = True

    def build_data(self):
        self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
        self.config.data.last_iter = self.state["last_iter"]

        root_dir, input_file = self.generate_custom_data()
        self.config.data.test.root_dir = root_dir
        self.config.data.test.meta_file = input_file

        # Fix: Change read_from from 'fake' to 'fs' for inference
        # 'fake' mode caches the first image and returns it for all subsequent calls
        # This causes all images to be identical
        original_read_from = self.config.data.get("read_from", "fs")
        if original_read_from == "fake":
            self.logger.warning(
                f"read_from is set to 'fake', which caches the first image. Changing to 'fs' for inference to ensure correct image loading."
            )
            self.config.data.read_from = "fs"

        # Note: We'll replace the sampler after dataloader is built
        # to avoid issues with build_sampler expecting a type
        if hasattr(self.config.data.test, "sampler"):
            original_sampler_type = self.config.data.test.sampler.get(
                "type", "distributed"
            )
            self.logger.info(
                f"Test sampler type: {original_sampler_type} (will be replaced with SequentialSampler after dataloader creation)"
            )

        # Debug: check meta file contents
        if self.dist.rank == 0 and os.path.exists(input_file):
            self.logger.info(f"Checking meta file: {input_file}")
            with open(input_file, "r") as f:
                lines = f.readlines()
                self.logger.info(f"Meta file has {len(lines)} entries")
                if len(lines) > 0:
                    import json

                    # Try to determine format: JSON (custom) or space-separated (ImageNet)
                    valid_entries = []
                    is_json_format = False
                    for idx, line in enumerate(lines[: min(10, len(lines))]):
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        # Try JSON format first (custom dataset)
                        try:
                            entry = json.loads(line)
                            if "filename" in entry:
                                valid_entries.append(entry)
                                is_json_format = True
                                continue
                        except json.JSONDecodeError:
                            pass

                        # Try ImageNet format (space-separated: filename label)
                        try:
                            parts = line.split()
                            if len(parts) >= 2:
                                valid_entries.append(
                                    {"filename": parts[0], "label": parts[1]}
                                )
                                continue
                        except Exception:
                            pass

                    if len(valid_entries) > 0:
                        format_type = (
                            "JSON" if is_json_format else "ImageNet (space-separated)"
                        )
                        self.logger.info(f"Meta file format: {format_type}")
                        self.logger.info(
                            f"First valid entry: {valid_entries[0].get('filename', 'N/A')}"
                        )
                        if len(valid_entries) > 1:
                            self.logger.info(
                                f"Second valid entry: {valid_entries[1].get('filename', 'N/A')}"
                            )
                            # Check for duplicates
                            filenames = [
                                entry.get("filename", "") for entry in valid_entries
                            ]
                            unique_filenames = set(filenames)
                            self.logger.info(
                                f"First {len(valid_entries)} valid entries: {len(unique_filenames)} unique filenames out of {len(filenames)}"
                            )
                            if len(unique_filenames) < len(filenames):
                                self.logger.warning(
                                    f"Found duplicate filenames in meta file!"
                                )
                    else:
                        self.logger.warning(
                            "No valid entries found in first 10 lines of meta file"
                        )

        if self.config.data.get("type", "imagenet") == "imagenet":
            self.val_data = build_imagenet_test_dataloader(self.config.data)
        else:
            self.val_data = build_custom_dataloader("test", self.config.data)

        # For inference, ensure sequential sampling to avoid duplicate samples
        # Replace any distributed/iterative sampler with SequentialSampler
        # Also set num_workers=0 to avoid multiprocessing issues that might cause duplicate data
        # Also wrap dataset to prevent modification of internal metas (which causes path duplication)
        if hasattr(self.val_data, "get") and "loader" in self.val_data:
            from torch.utils.data import SequentialSampler

            original_dataset = self.val_data["loader"].dataset

            # Wrap dataset to prevent modification of internal metas
            # ImageNetDataset.__getitem__ modifies self.metas[idx]['filename'], causing path duplication
            class SafeDatasetWrapper:
                def __init__(self, dataset):
                    self.dataset = dataset
                    # Create a deep copy of metas to prevent modification
                    import copy

                    if hasattr(dataset, "metas"):
                        # Deep copy all metas to prevent modification
                        self.original_metas = [
                            copy.deepcopy(meta) for meta in dataset.metas
                        ]
                    else:
                        self.original_metas = None

                def __len__(self):
                    return len(self.dataset)

                def __getitem__(self, idx):
                    # Before accessing, restore the original meta to prevent path duplication
                    if self.original_metas is not None and idx < len(
                        self.original_metas
                    ):
                        # Restore original meta (which may have been modified by previous access)
                        import copy

                        self.dataset.metas[idx] = copy.deepcopy(
                            self.original_metas[idx]
                        )
                    return self.dataset[idx]

                def __getattr__(self, name):
                    # Delegate all other attributes to the original dataset
                    return getattr(self.dataset, name)

            # Wrap the dataset
            safe_dataset = SafeDatasetWrapper(original_dataset)

            original_sampler = self.val_data["loader"].sampler
            original_num_workers = self.config.data.num_workers
            if original_sampler is not None:
                self.logger.info(
                    f"Replacing sampler {type(original_sampler).__name__} with SequentialSampler for inference"
                )
            sequential_sampler = SequentialSampler(safe_dataset)
            # Recreate DataLoader with sequential sampler and num_workers=0 to avoid multiprocessing issues
            # num_workers > 0 can sometimes cause issues with data loading, especially with custom datasets
            self.val_data["loader"] = torch.utils.data.DataLoader(
                dataset=safe_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=False,  # No shuffle for sequential
                sampler=sequential_sampler,
                num_workers=0,  # Set to 0 to avoid multiprocessing issues
                pin_memory=False,  # Disable pin_memory when num_workers=0
            )
            self.logger.info(
                f"DataLoader reconfigured with SequentialSampler and SafeDatasetWrapper for inference (num_workers changed from {original_num_workers} to 0)"
            )

    def generate_custom_data(self):

        if self.meta_file != "" and os.path.exists(self.meta_file):
            # If meta_file is provided, check if filenames in it are absolute paths or already include root_dir
            # If so, we need to adjust root_dir to avoid path duplication
            image_dir = self.image_dir
            # Check first few lines to see if filenames are absolute or relative
            try:
                with open(self.meta_file, "r") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        import json

                        # Try JSON format
                        try:
                            entry = json.loads(first_line)
                            filename = entry.get("filename", "")
                        except:
                            # Try ImageNet format
                            parts = first_line.split()
                            filename = parts[0] if len(parts) > 0 else ""

                        # Normalize paths for comparison
                        image_dir_norm = os.path.normpath(image_dir)
                        filename_norm = os.path.normpath(filename)

                        # If filename is absolute, set root_dir to empty
                        if os.path.isabs(filename_norm):
                            image_dir = ""
                            self.logger.info(
                                f"Filename is absolute, setting root_dir to empty"
                            )
                        # If filename already starts with image_dir, set root_dir to empty to avoid duplication
                        elif filename_norm.startswith(image_dir_norm):
                            image_dir = ""
                            self.logger.info(
                                f"Filename already includes root_dir, setting root_dir to empty to avoid path duplication"
                            )
                        else:
                            self.logger.info(
                                f"Filename is relative, using image_dir: {image_dir}"
                            )
                        self.logger.info(
                            f"Using provided meta_file: {self.meta_file}, adjusted image_dir: {image_dir}"
                        )
            except Exception as e:
                self.logger.warning(
                    f"Could not check meta_file format: {e}, using original image_dir"
                )

            return image_dir, self.meta_file

        input_file = os.path.join(self.output, "tmp_meta.json")
        image_dir = self.image_dir

        if os.path.isfile(self.image_dir):
            image_dir = os.path.abspath(os.path.dirname(self.image_dir))
            if self.dist.rank == 0:
                with open(input_file, "w") as output:
                    output.write(
                        json.dumps(
                            {"filename": os.path.basename(self.image_dir)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        else:
            if self.dist.rank == 0:
                with open(input_file, "w") as output:
                    meta_list = []
                    for root, dirs, files in os.walk(self.image_dir, topdown=False):
                        for name in files:
                            abs_path = os.path.join(root, name)
                            meta_list.append(
                                abs_path[len(self.image_dir) :].lstrip("/")
                            )
                    sample_num = len(meta_list)
                    if 0 < self.sample < 1:
                        sample_num = int(self.sample * sample_num)
                    elif self.sample > 1:
                        sample_num = min(
                            sample_num, int(self.sample)
                        )  # Fix: should be min, not max

                    self.logger.info(
                        f"Generating meta file with {sample_num} samples from {len(meta_list)} total files"
                    )
                    for idx in range(sample_num):
                        if idx >= len(meta_list):
                            self.logger.warning(
                                f"Index {idx} exceeds meta_list length {len(meta_list)}, skipping"
                            )
                            break
                        output.write(
                            json.dumps(
                                {
                                    "filename": meta_list[idx],
                                    "label_name": "abs",
                                    "label": 1,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

        link.barrier()
        return image_dir, input_file

    def paint(self, filename, pred, label, outdir):

        num = len(filename)

        for idx in range(num):
            ax, fig, h, w = self.get_axis(filename[idx])
            self.paint_one_image(pred[idx], label[idx], ax, h, w)
            out_name = os.path.join(outdir, os.path.basename(filename[idx]))
            fig.savefig(out_name, dpi=200)
            plt.close("all")

    @staticmethod
    def paint_one_image(pred, label, ax, h, w):
        font_sz = max(min(np.log(h) / np.log(100), np.log(w) / np.log(100)), 1)
        x1 = w // 8
        y1 = h // 8

        ax.text(
            x1,
            y1,
            f"cls {int(label)}, score:{pred[int(label)]:.3f}",
            fontsize=font_sz + 10,
            family="serif",
            color="r",
        )

    @staticmethod
    def get_axis(img_path):
        # Fix: handle path duplication issue
        # If path is duplicated (e.g., datasets/images/val/datasets/images/val/...), try to fix it
        if not os.path.exists(img_path):
            # Try to remove duplicate path segments
            parts = img_path.split(os.sep)
            # Look for repeated patterns
            for i in range(1, len(parts) // 2 + 1):
                if parts[:i] == parts[i : 2 * i]:
                    # Found duplicate, remove the first occurrence
                    fixed_path = os.sep.join(parts[i:])
                    if os.path.exists(fixed_path):
                        img_path = fixed_path
                        break
            # If still not found, try with absolute path
            if not os.path.exists(img_path) and not os.path.isabs(img_path):
                abs_path = os.path.abspath(img_path)
                if os.path.exists(abs_path):
                    img_path = abs_path

        assert os.path.exists(
            img_path
        ), f"check img file path, {img_path} (original: {img_path})"
        img = cv2.imread(img_path)[:, :, (2, 1, 0)]

        fig = plt.figure(frameon=False)
        fig.set_size_inches(img.shape[1] / 200, img.shape[0] / 200)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        fig.add_axes(ax)
        ax.imshow(img)
        return ax, fig, img.shape[0], img.shape[1]

    def inference(self):
        self.model.eval()

        res_file = os.path.join(self.output, f"results.txt.rank{self.dist.rank}")
        writer = open(res_file, "w")

        # Debug: check dataset size (but don't access items directly as it modifies internal state)
        if hasattr(self.val_data["loader"], "dataset"):
            dataset = self.val_data["loader"].dataset
            self.logger.info(f"Dataset size: {len(dataset)}")
            # Note: We don't access dataset[i] directly here because ImageNetDataset.__getitem__
            # modifies self.metas[idx]['filename'], which causes path duplication on subsequent accesses.
            # Instead, we'll check the data from the DataLoader batches below.

        for batch_idx, batch in enumerate(self.val_data["loader"]):
            if batch_idx > 0:
                break

            # Debug: check raw batch data before processing
            if "filename" in batch:
                print(f"Batch {batch_idx}: filenames count = {len(batch['filename'])}")
                if len(batch["filename"]) > 0:
                    print(f"First filename: {batch['filename'][0]}")
                    if len(batch["filename"]) > 1:
                        print(f"Second filename: {batch['filename'][1]}")
                        print(
                            f"Filenames are identical: {batch['filename'][0] == batch['filename'][1]}"
                        )
                        # Check if all filenames are the same
                        all_same = all(
                            f == batch["filename"][0] for f in batch["filename"]
                        )
                        print(f"All filenames are identical: {all_same}")

            # Debug: check raw image data before normalization/transforms
            raw_input = batch["image"]
            print(f"Batch {batch_idx}: raw input shape {raw_input.shape}")
            print(f"Raw input dtype: {raw_input.dtype}")
            if raw_input.size(0) > 1:
                print(
                    f"Raw first sample std: {raw_input[0].std().item():.6f}, mean: {raw_input[0].mean().item():.6f}"
                )
                print(
                    f"Raw second sample std: {raw_input[1].std().item():.6f}, mean: {raw_input[1].mean().item():.6f}"
                )
                print(
                    f"Raw samples are identical: {torch.allclose(raw_input[0], raw_input[1], atol=1e-6)}"
                )
                # Check pixel-level differences
                diff = (raw_input[0] - raw_input[1]).abs().max().item()
                print(f"Max pixel difference between first two samples: {diff}")

            input = batch["image"]
            input = input.cuda()

            # Debug: check if all inputs are the same after cuda
            print(f"Batch {batch_idx}: input shape {input.shape}")
            print(
                f"Input std: {input.std().item():.6f}, mean: {input.mean().item():.6f}"
            )
            if input.size(0) > 1:
                print(
                    f"First sample std: {input[0].std().item():.6f}, mean: {input[0].mean().item():.6f}"
                )
                print(
                    f"Second sample std: {input[1].std().item():.6f}, mean: {input[1].mean().item():.6f}"
                )
                print(
                    f"Samples are identical: {torch.allclose(input[0], input[1], atol=1e-6)}"
                )
                # Check pixel-level differences
                diff = (input[0] - input[1]).abs().max().item()
                print(
                    f"Max pixel difference between first two samples (after cuda): {diff}"
                )

            # compute output
            logits = self.model(input)

            # Debug: check logits
            print(f"Logits shape: {logits.shape}")
            print(
                f"Logits std: {logits.std().item():.6f}, mean: {logits.mean().item():.6f}"
            )
            if logits.size(0) > 1:
                print(
                    f"First logit: {logits[0, :5].detach().cpu().numpy()}"
                )  # Show first 5 classes
                print(f"Second logit: {logits[1, :5].detach().cpu().numpy()}")
                print(
                    f"Logits are identical: {torch.allclose(logits[0], logits[1], atol=1e-6)}"
                )

            scores = F.softmax(logits, dim=1)
            # compute prediction
            _, preds = logits.data.topk(k=1, dim=1)
            preds = preds.view(-1)
            # update batch information
            batch.update({"prediction": preds.detach()})
            batch.update({"score": scores.detach()})
            # save prediction information
            if self.cam:
                heatmap = self.gradCam(input)
                for idx in range(len(heatmap)):
                    basename = os.path.basename(batch["filename"][idx])
                    ext = basename.split(".")[-1]
                    basename = basename.replace("." + ext, "_cam" + "." + ext)

                    heatmap[idx].save(os.path.join(self.output, basename))

            if self.attn_rollout:
                rollout_maps = self.attentionRollout(input)
                for idx in range(len(rollout_maps)):
                    basename = os.path.basename(batch["filename"][idx])
                    ext = basename.split(".")[-1]
                    basename = basename.replace("." + ext, "_rollout" + "." + ext)

                    rollout_maps[idx].save(os.path.join(self.output, basename))

            if self.visualize:
                self.paint(batch["filename"], scores, preds, self.output)
            self.val_data["loader"].dataset.dump(writer, batch)

        writer.close()
        link.barrier()

        return

    def save_feature(self, module, input, output):
        self.feature = output

    def save_gradient(self, module, grad_in, grad_out):

        self.gradient = grad_out[0].detach()

    def gradCam(self, x):
        print(f"gradCam input shape: {x.shape}")
        print(f"gradCam input std: {x.std().item():.6f}, mean: {x.mean().item():.6f}")
        if x.size(0) > 1:
            print(
                f"First sample in gradCam: std={x[0].std().item():.6f}, mean={x[0].mean().item():.6f}"
            )
            print(
                f"Second sample in gradCam: std={x[1].std().item():.6f}, mean={x[1].mean().item():.6f}"
            )
            print(f"gradCam inputs identical: {torch.allclose(x[0], x[1], atol=1e-6)}")

        model = self.model.eval()
        image_size = (x.size(-1), x.size(-2))
        heat_maps = []
        # ImageNet normalization parameters (C, 1, 1) for broadcasting with (C, H, W)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        # 找到目标模块
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.feature_name:
                target_module = module
                break

        if target_module is None:
            raise ValueError(f"Module {self.feature_name} not found in model")

        # 逐个处理每个样本，确保完全独立
        for i in range(x.size(0)):
            # 为每个样本创建新的存储对象和hook函数
            feature_storage = [None]  # 使用列表避免闭包问题
            gradient_storage = [None]

            def save_feature_hook(module, input, output):
                feature_storage[0] = output.detach().clone()

            def save_gradient_hook(module, grad_in, grad_out):
                if grad_out[0] is not None:
                    gradient_storage[0] = grad_out[0].detach().clone()

            # 注册 hook
            forward_handle = target_module.register_forward_hook(save_feature_hook)
            backward_handle = target_module.register_backward_hook(save_gradient_hook)

            try:
                # 为每个样本创建完全独立的输入
                single_input = x[i : i + 1].clone().detach().requires_grad_(True)

                # 确保模型处于 eval 模式并清零所有梯度
                model.eval()
                model.zero_grad()
                if single_input.grad is not None:
                    single_input.grad.zero_()

                # 前向传播
                output = model(single_input)

                # 获取预测类别
                classes = F.softmax(output, dim=1)
                score, class_idx = classes.max(dim=-1)

                # 反向传播，只对预测类别进行反向
                score.backward(retain_graph=False)

                # 检查特征和梯度是否成功获取
                if feature_storage[0] is None:
                    raise RuntimeError(f"Failed to capture feature for sample {i}")
                if gradient_storage[0] is None:
                    raise RuntimeError(f"Failed to capture gradient for sample {i}")

                feature = feature_storage[0]  # Shape: (B, ...)
                gradient = gradient_storage[0]  # Shape: (B, ...)

                # 判断是 CNN (4D) 还是 ViT (3D)
                # ViT: (B, N, D) where N is number of patches, D is hidden dim
                # CNN: (B, C, H, W) where C is channels, H/W are spatial dims
                is_vit = feature.dim() == 3  # ViT: (B, N, D), CNN: (B, C, H, W)

                # Store is_vit for later use in visualization
                is_vit_flag = is_vit

                if is_vit:
                    # ViT: feature shape is (B, N, D) where N is number of patches (+ 1 for cls_token)
                    # Remove batch dimension for single sample
                    if feature.dim() == 3 and feature.size(0) == 1:
                        feature = feature.squeeze(0)  # (N, D)
                        gradient = gradient.squeeze(0)  # (N, D)

                    # Check if there's a cls_token (first token)
                    # For ViT with cls_token, we need to exclude it
                    # cls_token is present if:
                    # 1. model uses 'token' classifier
                    # 2. AND we're hooking a layer inside transformer (before final classification)
                    has_cls_token = False
                    num_patches = feature.size(0)

                    # More reliable check: if model uses token classifier and we're in transformer
                    # Check both self.model and self.model.module (in case model is wrapped)
                    model_to_check = self.model
                    if hasattr(self.model, "module"):
                        model_to_check = self.model.module

                    has_token_classifier = False
                    if hasattr(model_to_check, "classifier"):
                        has_token_classifier = model_to_check.classifier == "token"

                    if has_token_classifier:
                        # Check if we're hooking a layer inside the transformer
                        # Layers inside transformer: anything with 'transformer', 'encoder', 'encoders'
                        # Layers outside: 'head', 'pre_logits', etc.
                        is_inside_transformer = any(
                            keyword in self.feature_name
                            for keyword in [
                                "transformer",
                                "encoder",
                                "encoders",
                                "norm1",
                                "norm2",
                                "attention",
                                "feedforward",
                            ]
                        )

                        if is_inside_transformer:
                            has_cls_token = True
                            # Remove cls_token (first token)
                            if feature.size(0) > 1:
                                original_size = feature.size(0)
                                feature = feature[
                                    1:
                                ]  # Remove cls_token, shape: (N_patches, D)
                                gradient = gradient[1:]  # Remove cls_token gradient
                                num_patches = feature.size(0)
                                if i == 0:  # Only print for first sample to avoid spam
                                    print(
                                        f"ViT Grad-CAM: Excluded cls_token (original size: {original_size}, after exclusion: {num_patches})"
                                    )

                    # For ViT Grad-CAM, we need to:
                    # 1. Compute importance weights for each patch by averaging gradients over feature dimension
                    # 2. Multiply features by weights and sum over feature dimension
                    # gradient: (N_patches, D), we want to get importance per patch
                    # Standard Grad-CAM for ViT: weight = mean(gradient over D), then mask = sum(weight * feature over D)

                    # Calculate weights: global average pooling of gradients over feature dimension
                    # This gives importance weight for each patch token
                    weight = gradient.mean(
                        dim=-1
                    )  # (N_patches,) - importance per patch

                    # Calculate Grad-CAM: for each patch, sum over feature dimension after weighting
                    # We can use the weight directly, or multiply with features
                    # Option 1: Use gradient-weighted features (more standard)
                    # mask = F.relu((weight.unsqueeze(-1) * feature).sum(dim=-1))  # (N_patches,)

                    # Option 2: Use weight directly (simpler, often works better for ViT)
                    # For ViT, the gradient already captures the importance, so we can use it directly
                    mask = F.relu(weight)  # (N_patches,)

                    # Alternative: Use both gradient and feature information
                    # This combines both the gradient importance and feature activation
                    # mask = F.relu((weight.unsqueeze(-1) * feature).mean(dim=-1))  # Average over features

                    # Reshape to 2D spatial layout
                    # Calculate number of patches per side
                    # For ViT-B/16 with 224x224 image: 14x14 = 196 patches
                    patches_per_side = int(np.sqrt(num_patches))

                    # Verify patch count matches expected number
                    # According to VisionTransformer structure:
                    # - num_patches = (image_size // patch_size) ** 2
                    # - For vit_b16_224: (224 // 16) ** 2 = 14 ** 2 = 196
                    # - If classifier == 'token', transformer has 197 tokens (1 cls_token + 196 patches)
                    # - After excluding cls_token, we should have 196 patches
                    if hasattr(model_to_check, "num_patches"):
                        expected_patches = model_to_check.num_patches
                        if num_patches != expected_patches:
                            if i == 0:
                                print(
                                    f"Warning: num_patches mismatch. Got {num_patches}, expected {expected_patches}. "
                                    f"Has cls_token been excluded? has_cls_token={has_cls_token}"
                                )
                                if (
                                    has_cls_token
                                    and num_patches == expected_patches + 1
                                ):
                                    print(
                                        f"  -> cls_token was NOT excluded! Excluding it now..."
                                    )
                                    # This shouldn't happen if code is correct, but handle it anyway
                                    feature = feature[1:]
                                    gradient = gradient[1:]
                                    num_patches = feature.size(0)
                                    patches_per_side = int(np.sqrt(num_patches))

                    if patches_per_side * patches_per_side == num_patches:
                        # Perfect square, reshape to 2D
                        mask = mask.reshape(patches_per_side, patches_per_side)
                        if i == 0:
                            print(
                                f"ViT Grad-CAM: Reshaped mask to {patches_per_side}x{patches_per_side} (total patches: {num_patches})"
                            )
                    else:
                        # Not a perfect square, try to infer from model
                        # Use model_to_check that was defined earlier
                        if hasattr(model_to_check, "num_patches"):
                            expected_patches = model_to_check.num_patches
                            patches_per_side = int(np.sqrt(expected_patches))
                            if (
                                patches_per_side * patches_per_side == expected_patches
                                and mask.size(0) >= expected_patches
                            ):
                                mask = mask[:expected_patches].reshape(
                                    patches_per_side, patches_per_side
                                )
                            else:
                                # Fallback: try to infer from image size and patch size
                                if hasattr(model_to_check, "patch_size") and hasattr(
                                    model_to_check, "num_patches"
                                ):
                                    patches_per_side = int(
                                        np.sqrt(model_to_check.num_patches)
                                    )
                                    if (
                                        patches_per_side * patches_per_side
                                        == model_to_check.num_patches
                                    ):
                                        mask = mask[
                                            : model_to_check.num_patches
                                        ].reshape(patches_per_side, patches_per_side)
                                    else:
                                        # Last resort: create a square mask
                                        side_len = int(np.sqrt(mask.size(0)))
                                        if side_len * side_len <= mask.size(0):
                                            mask = mask[: side_len * side_len].reshape(
                                                side_len, side_len
                                            )
                                        else:
                                            mask = mask.unsqueeze(0)  # (1, N)
                                else:
                                    mask = mask.unsqueeze(0)  # (1, N)
                        else:
                            # Fallback: create a square mask
                            side_len = int(np.sqrt(mask.size(0)))
                            if side_len * side_len <= mask.size(0):
                                mask = mask[: side_len * side_len].reshape(
                                    side_len, side_len
                                )
                            else:
                                mask = mask.unsqueeze(0)  # (1, N)

                    mask_np = mask.cpu().numpy().astype(np.float32)

                else:
                    # CNN: feature shape is (B, C, H, W)
                    # 计算权重：对空间维度求平均
                    if gradient.dim() == 4:
                        weight = gradient.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
                    else:
                        weight = gradient.mean(dim=(-1, -2), keepdim=True)

                    # 计算 Grad-CAM mask
                    if feature.dim() == 4:
                        mask = F.relu((weight * feature).sum(dim=1)).squeeze(
                            0
                        )  # (H, W)
                    else:
                        mask = F.relu((weight * feature).sum(dim=1))
                        if mask.dim() > 2:
                            mask = mask.squeeze(0)

                    mask_np = mask.cpu().numpy().astype(np.float32)
                    if mask_np.ndim == 0:
                        mask_np = mask_np.reshape(1, 1)
                    elif mask_np.ndim == 1:
                        # 如果是一维，需要 reshape 成 2D
                        side_len = int(np.sqrt(mask_np.shape[0]))
                        if side_len * side_len == mask_np.shape[0]:
                            mask_np = mask_np.reshape(side_len, side_len)
                        else:
                            mask_np = mask_np.reshape(1, -1)

                # 反归一化图像以匹配原始图像（用于可视化）
                img = single_input[0].detach().cpu().numpy()
                # 确保 img 是 (C, H, W) 形状
                if img.ndim == 4:
                    img = img.squeeze(0)
                assert (
                    img.ndim == 3 and img.shape[0] == 3
                ), f"Expected img shape (3, H, W), got {img.shape}"
                img = img * std + mean  # 反归一化
                img = np.clip(img, 0, 1)  # 裁剪到 [0, 1] 范围

                # 将 mask 调整到图像大小
                if mask_np.ndim == 2:
                    mask_resized = cv2.resize(mask_np, image_size)
                else:
                    # Fallback: if still 1D, try to reshape
                    side_len = int(np.sqrt(mask_np.size))
                    if side_len * side_len == mask_np.size:
                        mask_np = mask_np.reshape(side_len, side_len)
                        mask_resized = cv2.resize(mask_np, image_size)
                    else:
                        # Last resort: create a dummy mask
                        mask_resized = np.ones(image_size[::-1], dtype=np.float32) * 0.5

                # 归一化 mask - 使用更激进的归一化来增强对比度
                mask_min = np.min(mask_resized)
                mask_max = np.max(mask_resized)
                if mask_max > mask_min:
                    # Standard normalization
                    mask_resized = (mask_resized - mask_min) / (mask_max - mask_min)

                    # For ViT, apply additional enhancement to make hotspots more visible
                    if is_vit_flag:
                        # Apply power scaling to enhance high-activation regions
                        mask_resized = np.power(
                            mask_resized, 0.7
                        )  # Gamma correction (0.7 makes it brighter)
                        # Re-normalize after power scaling
                        mask_min = np.min(mask_resized)
                        mask_max = np.max(mask_resized)
                        if mask_max > mask_min:
                            mask_resized = (mask_resized - mask_min) / (
                                mask_max - mask_min
                            )
                else:
                    mask_resized = np.zeros_like(mask_resized)

                # 生成热力图
                heat_map = np.float32(
                    cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
                )

                # 将图像从 CHW 转换为 HWC，并转换为 uint8
                img_hwc = (img.transpose((1, 2, 0)) * 255).astype(np.uint8)
                # 将 RGB 转换为 BGR（因为 OpenCV 的 applyColorMap 输出是 BGR 格式）
                img_hwc = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)

                # 混合热力图和原始图像 - 对于 ViT 使用更高的热力图权重
                if is_vit_flag:
                    # For ViT, use higher weight for heatmap to make it more visible
                    cam = heat_map * 0.5 + np.float32(img_hwc) * 0.5
                else:
                    # For CNN, use original weights
                    cam = heat_map * 0.4 + np.float32(img_hwc) * 0.6
                cam = np.clip(cam, 0, 255).astype(np.uint8)

                # 将 BGR 转换为 RGB 用于 PIL
                cam_rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
                heat_maps.append(transforms.ToPILImage()(cam_rgb))

            finally:
                # 确保 hook 被清除
                forward_handle.remove()
                backward_handle.remove()

                # 清理梯度
                if single_input.grad is not None:
                    single_input.grad = None
                model.zero_grad()
                # 清理存储
                feature_storage[0] = None
                gradient_storage[0] = None

        return heat_maps

    def attentionRollout(self, x):
        """
        Compute attention rollout for ViT models.
        Attention rollout aggregates attention weights across all transformer layers
        to show which image patches the model focuses on.

        Reference: https://arxiv.org/abs/2005.00928
        """
        model = self.model.eval()
        image_size = (x.size(-1), x.size(-2))
        rollout_maps = []

        # ImageNet normalization parameters
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        # Check if model is ViT
        model_to_check = self.model
        if hasattr(self.model, "module"):
            model_to_check = self.model.module

        is_vit = False
        if hasattr(model_to_check, "transformer") and hasattr(
            model_to_check, "classifier"
        ):
            is_vit = True

        if not is_vit:
            raise ValueError(
                "Attention rollout is only supported for ViT models. "
                "Current model does not appear to be a ViT."
            )

        # Get transformer encoder
        if hasattr(self.model, "module"):
            transformer = self.model.module.transformer
        else:
            transformer = model_to_check.transformer

        # Find all encoder blocks
        encoder_blocks = []
        if hasattr(transformer, "encoders"):
            for i, encoder_block in enumerate(transformer.encoders):
                encoder_blocks.append((f"encoder_{i}", encoder_block))

        if len(encoder_blocks) == 0:
            raise ValueError("Could not find encoder blocks in transformer")

        # Process each sample
        for i in range(x.size(0)):
            single_input = x[i : i + 1].clone().detach()

            # Storage for attention weights from each layer
            attention_maps = []
            original_forwards = {}

            # Temporarily modify attention forward to capture weights
            for name, encoder_block in encoder_blocks:
                attention_module = encoder_block.attention

                # Create a wrapper that captures attention
                original_forward = attention_module.forward
                original_forwards[name] = original_forward

                # Create closure with proper variable capture
                def make_attention_capture(attn_module, attn_maps_list):
                    def forward_with_capture(x):
                        # Compute attention based on forward_type
                        if attn_module.forward_type == "einsum":
                            from einops import rearrange

                            qkv = attn_module.to_qkv(x)
                            q, k, v = rearrange(
                                qkv,
                                "b n (qkv h d) -> qkv b h n d",
                                qkv=3,
                                h=attn_module.heads,
                            )
                            dots = (
                                torch.einsum("bhid,bhjd->bhij", q, k)
                                * attn_module.scale
                            )
                            attn = dots.softmax(dim=-1)
                            attn = attn_module.attention_dropout(attn)
                            out = torch.einsum("bhij,bhjd->bhid", attn, v)
                            out = rearrange(out, "b h n d -> b n (h d)")
                            out = attn_module.to_out(out)
                            out = attn_module.dropout(out)
                        else:
                            B, N, C = x.shape
                            qkv = (
                                attn_module.to_qkv(x)
                                .reshape(
                                    B, N, 3, attn_module.heads, C // attn_module.heads
                                )
                                .permute(2, 0, 3, 1, 4)
                            )
                            q, k, v = qkv[0], qkv[1], qkv[2]
                            attn = (q @ k.transpose(-2, -1)) * attn_module.scale
                            attn = attn.softmax(dim=-1)
                            attn = attn_module.attention_dropout(attn)
                            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
                            out = attn_module.to_out(out)
                            out = attn_module.dropout(out)

                        # Store attention weights (average over heads)
                        attn_avg = attn.mean(dim=1)  # Average over heads: (B, N, N)
                        attn_maps_list.append(attn_avg[0].detach().cpu())  # (N, N)

                        return out

                    return forward_with_capture

                attention_module.forward = make_attention_capture(
                    attention_module, attention_maps
                )

            try:
                # Forward pass to collect attention weights
                with torch.no_grad():
                    output = model(single_input)

                # Restore original forwards
                for name, encoder_block in encoder_blocks:
                    attention_module = encoder_block.attention
                    if name in original_forwards:
                        attention_module.forward = original_forwards[name]

                if len(attention_maps) == 0:
                    raise RuntimeError("Failed to capture attention weights")

                # Compute attention rollout
                # Start with identity matrix
                rollout = torch.eye(attention_maps[0].size(0))

                # Rollout from last layer to first
                for attn in reversed(attention_maps):
                    # Add residual connection: I + attention
                    # Then normalize
                    rollout = rollout + attn
                    rollout = rollout / rollout.sum(dim=-1, keepdim=True)
                    rollout = rollout @ attn

                # Extract attention to cls_token (first token)
                # rollout shape: (N, N) where N = num_patches + 1 (with cls_token)
                if (
                    hasattr(model_to_check, "classifier")
                    and model_to_check.classifier == "token"
                ):
                    # Get attention from cls_token to all patches
                    cls_attention = rollout[
                        0, 1:
                    ]  # Skip cls_token itself, get attention to patches
                else:
                    # For GAP, use mean attention
                    cls_attention = (
                        rollout.mean(dim=0)[1:]
                        if rollout.size(0) > 1
                        else rollout.mean(dim=0)
                    )

                # Reshape to 2D spatial layout
                num_patches = cls_attention.size(0)
                patches_per_side = int(np.sqrt(num_patches))

                if patches_per_side * patches_per_side == num_patches:
                    mask = cls_attention.reshape(
                        patches_per_side, patches_per_side
                    ).numpy()
                else:
                    # Fallback
                    side_len = int(np.sqrt(num_patches))
                    if side_len * side_len <= num_patches:
                        mask = (
                            cls_attention[: side_len * side_len]
                            .reshape(side_len, side_len)
                            .numpy()
                        )
                    else:
                        mask = cls_attention.unsqueeze(0).numpy()

                # Resize to image size
                mask_resized = cv2.resize(mask.astype(np.float32), image_size)

                # Normalize
                mask_min = np.min(mask_resized)
                mask_max = np.max(mask_resized)
                if mask_max > mask_min:
                    mask_resized = (mask_resized - mask_min) / (mask_max - mask_min)
                else:
                    mask_resized = np.zeros_like(mask_resized)

                # Generate heatmap
                heat_map = np.float32(
                    cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
                )

                # Prepare image
                img = single_input[0].detach().cpu().numpy()
                if img.ndim == 4:
                    img = img.squeeze(0)
                assert (
                    img.ndim == 3 and img.shape[0] == 3
                ), f"Expected img shape (3, H, W), got {img.shape}"
                img = img * std + mean
                img = np.clip(img, 0, 1)
                img_hwc = (img.transpose((1, 2, 0)) * 255).astype(np.uint8)
                img_hwc = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)

                # Blend
                cam = heat_map * 0.5 + np.float32(img_hwc) * 0.5
                cam = np.clip(cam, 0, 255).astype(np.uint8)
                cam_rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
                rollout_maps.append(transforms.ToPILImage()(cam_rgb))

            except Exception as e:
                # Restore original forwards in case of error
                for name, encoder_block in encoder_blocks:
                    attention_module = encoder_block.attention
                    if name in original_forwards:
                        attention_module.forward = original_forwards[name]
                raise e

        return rollout_maps


@link_dist
def main():
    parser = argparse.ArgumentParser(description="Inference Solver")
    parser.add_argument("--config", required=True, type=str, help="Prototype task yaml")
    parser.add_argument("--recover", default="", help="Recover model path to visuazlie")

    parser.add_argument(
        "-i",
        "--image_dir",
        required=True,
        dest="image_dir",
        type=str,
        help="The image dir that you want to visuazlie.",
    )
    parser.add_argument(
        "-m",
        "--meta_file",
        required=False,
        dest="meta_file",
        type=str,
        help="The prototype custom meta file that you want to visuazlie. "
        "If this argument are not provide, we will visualize the images in {image_dir}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./inference_resuts",
        dest="output",
        help="the folder where results file or images will be saved.",
    )
    parser.add_argument(
        "--visualize",
        default=True,
        help="Whether paint class and score on images to visualize.",
    )
    parser.add_argument(
        "--sample",
        default=-1,
        type=float,
        help="if gived number -1, remain all results. if 0 < gived number <=1, "
        "sample {gived number * len(images_dir)} images, "
        "if gived number > 1, sample gived number images.",
    )
    parser.add_argument(
        "--cam",
        default=False,
        help="Whether save gradcam results. See https://arxiv.org/abs/1610.02391 for details.",
    )
    parser.add_argument(
        "--attn_rollout",
        default=False,
        action="store_true",
        help="Whether save attention rollout results for ViT models. "
        "Attention rollout aggregates attention weights across all transformer layers.",
    )
    parser.add_argument(
        "--name",
        default="module.layer4",
        help="the last feature extractor layer name you want to visualize gradcam results, "
        "e.g. 'layer4' in resnet series.",
    )

    args = parser.parse_args()
    # build solver
    inference_helper = Inference(args.__dict__)
    inference_helper.inference()


if __name__ == "__main__":
    main()
