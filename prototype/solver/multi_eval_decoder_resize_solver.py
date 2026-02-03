import os
import argparse
import sys
import subprocess
import numpy as np
from easydict import EasyDict
from tensorboardX import SummaryWriter
import time
import datetime
import torch
import random
import json
import prototype.spring.linklink as link
import torch.nn.functional as F
from copy import deepcopy

from .base_solver import BaseSolver
from prototype.prototype.utils.dist import link_dist, DistModule, broadcast_object
from prototype.prototype.utils.misc import (
    makedir,
    create_logger,
    get_logger,
    count_params,
    count_flops,
    param_group_all,
    AverageMeter,
    accuracy,
    load_state_model,
    load_state_optimizer,
    mixup_data,
    mix_criterion,
    cutmix_data,
    parse_config,
)
from prototype.prototype.utils.ema import EMA
from prototype.prototype.model import model_entry
from prototype.prototype.optimizer import optim_entry

# FP16 optimizers are optional and may not be available
try:
    from prototype.prototype.optimizer import (
        FP16RMSprop,
        FP16SGD,
        FusedFP16SGD,
        FP16AdamW,
    )
except ImportError:
    FP16RMSprop = None
    FP16SGD = None
    FusedFP16SGD = None
    FP16AdamW = None
from prototype.prototype.lr_scheduler import scheduler_entry
from prototype.prototype.data import (
    build_imagenet_train_dataloader,
    build_imagenet_test_dataloader,
)
from prototype.prototype.data import build_custom_dataloader
from prototype.prototype.loss_functions import LabelSmoothCELoss

# send_info is optional and may not be available
try:
    from prototype.prototype.utils.user_analysis_helper import send_info
except ImportError:

    def send_info(info):
        pass  # No-op if not available


# SPRING_MODELS_REGISTRY is optional and may not be available
try:
    from spring.models import SPRING_MODELS_REGISTRY
except ImportError:
    try:
        from prototype.spring.models import SPRING_MODELS_REGISTRY
    except ImportError:
        SPRING_MODELS_REGISTRY = None


class MultiEvalSolver_S(BaseSolver):

    def __init__(self, config, model=None, prefix_name=None):
        self.prototype_info = EasyDict()
        self.config = config
        if prefix_name is not None:
            self.prefix_name = prefix_name
        else:
            self.prefix_name = self.config.model.type

        self.setup_env()

        if model is not None:
            self.model = model
            # Move model to the correct GPU for this rank
            device_id = self.dist.rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            self.model.cuda(device=device_id)
            # Wrap with DistModule for distributed support
            if not isinstance(self.model, DistModule):
                self.model = DistModule(self.model, self.config.dist.sync)
        else:
            self.build_model()
        # self.build_optimizer()
        self.build_data()
        # self.build_lr_scheduler()
        send_info(self.prototype_info)
        self.logger.info(f"{self.prefix_name} start!")

    def setup_env(self):
        # dist
        self.dist = EasyDict()
        self.dist.rank, self.dist.world_size = link.get_rank(), link.get_world_size()
        self.prototype_info.world_size = self.dist.world_size
        # directories
        self.path = EasyDict()
        self.path.root_path = os.getcwd()
        self.path.save_path = os.path.join(self.path.root_path, "checkpoints")
        self.path.event_path = os.path.join(self.path.root_path, "events")
        self.path.result_path = os.path.join(
            self.path.root_path, self.prefix_name, "results"
        )
        makedir(self.path.save_path)
        makedir(self.path.event_path)
        makedir(self.path.result_path)
        # tb_logger
        if self.dist.rank == 0:
            self.tb_logger = SummaryWriter(self.path.event_path)
        # logger
        create_logger(os.path.join(self.path.root_path, self.prefix_name, "log.txt"))
        self.logger = get_logger(__name__)
        # self.logger.info(f'config: {pprint.pformat(self.config)}')
        if "SLURM_NODELIST" in os.environ:
            self.logger.info(f"hostnames: {os.environ['SLURM_NODELIST']}")
        # load pretrain checkpoint
        if hasattr(self.config.saver, "pretrain"):
            self.state = torch.load(self.config.saver.pretrain.path, "cpu", weights_only=False)
            # self.logger.info(
            #    f"Recovering from {self.config.saver.pretrain.path}, keys={list(self.state.keys())}"
            # )
            if hasattr(self.config.saver.pretrain, "ignore"):
                from prototype.prototype.utils.misc import modify_state

                self.state = modify_state(self.state, self.config.saver.pretrain.ignore)
            # Ensure last_iter exists (some checkpoints may not have it)
            if "last_iter" not in self.state:
                self.state["last_iter"] = 0
        else:
            self.state = {}
            self.state["last_iter"] = 0
        # others
        torch.backends.cudnn.benchmark = True

    def build_model(self):
        if hasattr(self.config, "lms"):
            if self.config.lms.enable:
                torch.cuda.set_enabled_lms(True)
                byte_limit = self.config.lms.kwargs.limit * (1 << 30)
                torch.cuda.set_limit_lms(byte_limit)
                self.logger.info(
                    "Enable large model support, limit of {}G!".format(
                        self.config.lms.kwargs.limit
                    )
                )

        self.model = model_entry(self.config.model, full_config=self.config)
        self.prototype_info.model = self.config.model.type
        # Move model to the correct GPU for this rank
        device_id = self.dist.rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        self.model.cuda(device=device_id)

        if self.dist.rank == 0:
            count_params(self.model)
            count_flops(
                self.model,
                input_shape=[
                    1,
                    3,
                    self.config.data.input_size,
                    self.config.data.input_size,
                ],
            )

        # handle fp16
        if (
            self.config.optimizer.type == "FP16SGD"
            or self.config.optimizer.type == "FusedFP16SGD"
            or self.config.optimizer.type == "FP16RMSprop"
            or self.config.optimizer.type == "FP16AdamW"
        ):
            self.fp16 = True
        else:
            self.fp16 = False

        if self.fp16:
            # if you have modules that must use fp32 parameters, and need fp32 input
            # try use link.fp16.register_float_module(your_module)
            # if you only need fp32 parameters set cast_args=False when call this
            # function, then call link.fp16.init() before call model.half()
            if self.config.optimizer.get("fp16_normal_bn", False):
                self.logger.info("using normal bn for fp16")
                link.fp16.register_float_module(
                    link.nn.SyncBatchNorm2d, cast_args=False
                )
                link.fp16.register_float_module(torch.nn.BatchNorm2d, cast_args=False)
            if self.config.optimizer.get("fp16_normal_fc", False):
                self.logger.info("using normal fc for fp16")
                link.fp16.register_float_module(torch.nn.Linear, cast_args=True)
            link.fp16.init()
            self.model.half()

        # Load checkpoint before wrapping with DistModule to avoid key mismatch
        if "model" in self.state:
            # Remove 'module.' prefix from state dict keys if present
            # This handles checkpoints saved with DataParallel or DistributedDataParallel
            model_state = (
                self.state["model"].copy()
                if isinstance(self.state["model"], dict)
                else self.state["model"]
            )
            if isinstance(model_state, dict):
                new_model_state = {}
                for key, value in model_state.items():
                    if "module." in key:
                        new_model_state[key.split("module.")[1]] = value
                    else:
                        new_model_state[key] = value
                model_state = new_model_state
            # Load state dict into unwrapped model
            self.model.load_state_dict(model_state, strict=False)

        # Synchronize all ranks before wrapping with DistModule
        if self.dist.world_size > 1:
            link.barrier()
        self.model = DistModule(self.model, self.config.dist.sync)

    def build_optimizer(self):

        opt_config = self.config.optimizer
        opt_config.kwargs.lr = self.config.lr_scheduler.kwargs.base_lr
        self.prototype_info.optimizer = self.config.optimizer.type

        # make param_groups
        pconfig = {}

        if opt_config.get("no_wd", False):
            pconfig["conv_b"] = {"weight_decay": 0.0}
            pconfig["linear_b"] = {"weight_decay": 0.0}
            pconfig["bn_w"] = {"weight_decay": 0.0}
            pconfig["bn_b"] = {"weight_decay": 0.0}

        if "pconfig" in opt_config:
            pconfig.update(opt_config["pconfig"])

        param_group, type2num = param_group_all(self.model, pconfig)

        opt_config.kwargs.params = param_group

        self.optimizer = optim_entry(opt_config)

        if "optimizer" in self.state:
            load_state_optimizer(self.optimizer, self.state["optimizer"])

        # EMA
        if self.config.ema.enable:
            self.config.ema.kwargs.model = self.model
            self.ema = EMA(**self.config.ema.kwargs)
        else:
            self.ema = None

        if "ema" in self.state:
            self.ema.load_state_dict(self.state["ema"])

    def build_lr_scheduler(self):
        self.prototype_info.lr_scheduler = self.config.lr_scheduler.type
        if not getattr(self.config.lr_scheduler.kwargs, "max_iter", False):
            self.config.lr_scheduler.kwargs.max_iter = self.config.data.max_iter
        self.config.lr_scheduler.kwargs.optimizer = (
            self.optimizer.optimizer
            if (FP16SGD is not None and isinstance(self.optimizer, FP16SGD))
            or (FP16RMSprop is not None and isinstance(self.optimizer, FP16RMSprop))
            or (FP16AdamW is not None and isinstance(self.optimizer, FP16AdamW))
            else self.optimizer
        )
        self.config.lr_scheduler.kwargs.last_iter = self.state["last_iter"]
        self.lr_scheduler = scheduler_entry(self.config.lr_scheduler)

    def build_data(self):
        self.config.data.last_iter = self.state["last_iter"]
        if getattr(self.config.lr_scheduler.kwargs, "max_iter", False):
            self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
        else:
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.max_epoch

        # Only load test/val data for evaluation, skip training data
        if self.config.data.get("type", "imagenet") == "imagenet":
            self.val_data = build_imagenet_test_dataloader(self.config.data)
        else:
            self.val_data = build_custom_dataloader("test", self.config.data)

    def pre_train(self):
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.config.saver.print_freq)
        self.meters.step_time = AverageMeter(self.config.saver.print_freq)
        self.meters.data_time = AverageMeter(self.config.saver.print_freq)
        self.meters.losses = AverageMeter(self.config.saver.print_freq)
        self.meters.top1 = AverageMeter(self.config.saver.print_freq)
        self.meters.top5 = AverageMeter(self.config.saver.print_freq)

        self.model.train()

        label_smooth = self.config.get("label_smooth", 0.0)
        self.num_classes = self.config.model.kwargs.get("num_classes", 1000)
        self.topk = 5 if self.num_classes >= 5 else self.num_classes
        if label_smooth > 0:
            self.logger.info("using label_smooth: {}".format(label_smooth))
            self.criterion = LabelSmoothCELoss(label_smooth, self.num_classes)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        self.mixup = self.config.get("mixup", 1.0)
        self.cutmix = self.config.get("cutmix", 0.0)
        self.switch_prob = 0.0
        if self.mixup < 1.0:
            self.logger.info("using mixup with alpha of: {}".format(self.mixup))
        if self.cutmix > 0.0:
            self.logger.info("using cutmix with alpha of: {}".format(self.cutmix))
        if self.mixup < 1.0 and self.cutmix > 0.0:
            # the probability of switching mixup to cutmix if both are activated
            self.switch_prob = self.config.get("switch_prob", 0.5)
            self.logger.info(
                "switching between mixup and cutmix with probility of: {}".format(
                    self.switch_prob
                )
            )

    def train(self):

        self.pre_train()
        total_step = len(self.train_data["loader"])
        start_step = self.state["last_iter"] + 1
        end = time.time()
        for i, batch in enumerate(self.train_data["loader"]):
            input = batch["image"]
            target = batch["label"]
            curr_step = start_step + i
            self.lr_scheduler.step(curr_step)
            # lr_scheduler.get_lr()[0] is the main lr
            current_lr = self.lr_scheduler.get_lr()[0]
            # measure data loading time
            self.meters.data_time.update(time.time() - end)
            # transfer input to gpu
            target = target.squeeze().cuda().long()
            input = input.cuda().half() if self.fp16 else input.cuda()
            # mixup
            if self.mixup < 1.0 and random.uniform(0, 1) > self.switch_prob:
                input, target_a, target_b, lam = mixup_data(input, target, self.mixup)
            # cutmix
            elif self.cutmix > 0.0:
                input, target_a, target_b, lam = cutmix_data(input, target, self.cutmix)
            # forward
            logits = self.model(input)
            # mixup
            if self.mixup < 1.0 or self.cutmix > 0.0:
                loss = mix_criterion(self.criterion, logits, target_a, target_b, lam)
                loss /= self.dist.world_size
            else:
                loss = self.criterion(logits, target) / self.dist.world_size
            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits, target, topk=(1, self.topk))

            reduced_loss = loss.clone()
            reduced_prec1 = prec1.clone() / self.dist.world_size
            reduced_prec5 = prec5.clone() / self.dist.world_size

            self.meters.losses.reduce_update(reduced_loss)
            self.meters.top1.reduce_update(reduced_prec1)
            self.meters.top5.reduce_update(reduced_prec5)

            # compute and update gradient
            self.optimizer.zero_grad()
            if FusedFP16SGD is not None and isinstance(self.optimizer, FusedFP16SGD):
                self.optimizer.backward(loss)
                self.model.sync_gradients()
                self.optimizer.step()
            elif (FP16SGD is not None and isinstance(self.optimizer, FP16SGD)) or (
                FP16RMSprop is not None and isinstance(self.optimizer, FP16RMSprop)
            ):

                def closure():
                    self.optimizer.backward(loss, False)
                    self.model.sync_gradients()
                    # check overflow, convert to fp32 grads, downscale
                    self.optimizer.update_master_grads()
                    return loss

                self.optimizer.step(closure)
            else:
                loss.backward()
                self.model.sync_gradients()
                self.optimizer.step()

            # EMA
            if self.ema is not None:
                self.ema.step(self.model, curr_step=curr_step)
            # measure elapsed time
            self.meters.batch_time.update(time.time() - end)

            # training logger
            if curr_step % self.config.saver.print_freq == 0 and self.dist.rank == 0:
                self.tb_logger.add_scalar(
                    "loss_train", self.meters.losses.avg, curr_step
                )
                self.tb_logger.add_scalar("acc1_train", self.meters.top1.avg, curr_step)
                self.tb_logger.add_scalar("acc5_train", self.meters.top5.avg, curr_step)
                self.tb_logger.add_scalar("lr", current_lr, curr_step)
                remain_secs = (total_step - curr_step) * self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs)
                )
                log_msg = (
                    f"Iter: [{curr_step}/{total_step}]\t"
                    f"Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t"
                    f"Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t"
                    f"Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t"
                    f"Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t"
                    f"Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t"
                    f"LR {current_lr:.4f}\t"
                    f"Remaining Time {remain_time} ({finish_time})"
                )
                self.logger.info(log_msg)

            # testing during training
            if curr_step > 0 and curr_step % self.config.saver.val_freq == 0:
                metrics = self.evaluate()
                if self.ema is not None:
                    self.ema.load_ema(self.model)
                    ema_metrics = self.evaluate()
                    self.ema.recover(self.model)
                    if (
                        self.dist.rank == 0
                        and self.config.data.test.evaluator.type == "imagenet"
                    ):
                        metric_key = "top{}".format(self.topk)
                        self.tb_logger.add_scalars(
                            "acc1_val", {"ema": ema_metrics.metric["top1"]}, curr_step
                        )
                        self.tb_logger.add_scalars(
                            "acc5_val",
                            {"ema": ema_metrics.metric[metric_key]},
                            curr_step,
                        )

                # testing logger
                if (
                    self.dist.rank == 0
                    and self.config.data.test.evaluator.type == "imagenet"
                ):
                    metric_key = "top{}".format(self.topk)
                    self.tb_logger.add_scalar(
                        "acc1_val", metrics.metric["top1"], curr_step
                    )
                    self.tb_logger.add_scalar(
                        "acc5_val", metrics.metric[metric_key], curr_step
                    )

                # save ckpt
                if self.dist.rank == 0:
                    if self.config.saver.save_many:
                        ckpt_name = f"{self.path.save_path}/ckpt_{curr_step}.pth.tar"
                    else:
                        ckpt_name = f"{self.path.save_path}/ckpt.pth.tar"
                    self.state["model"] = self.model.state_dict()
                    self.state["optimizer"] = self.optimizer.state_dict()
                    self.state["last_iter"] = curr_step
                    if self.ema is not None:
                        self.state["ema"] = self.ema.state_dict()
                    torch.save(self.state, ckpt_name)

            end = time.time()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        # Ensure all ranks are synchronized before evaluation
        if self.dist.world_size > 1:
            link.barrier()
        imagenetc_flag = self.config.data.test.get("imagenet_c", False)
        if imagenetc_flag:

            noise_list = []

            writer = {
                "noise": {"gaussian_noise": {}, "shot_noise": {}, "impulse_noise": {}},
                "blur": {
                    "defocus_blur": {},
                    "glass_blur": {},
                    "motion_blur": {},
                    "zoom_blur": {},
                },
                "weather": {"snow": {}, "frost": {}, "fog": {}, "brightness": {}},
                "digital": {
                    "contrast": {},
                    "elastic_transform": {},
                    "pixelate": {},
                    "jpeg_compression": {},
                },
                "extra": {
                    "speckle_noise": {},
                    "spatter": {},
                    "gaussian_blur": {},
                    "saturate": {},
                },
            }
            for noise in writer:
                for noise_type in writer[noise]:
                    for i in range(1, 6):
                        res_file = os.path.join(
                            self.path.result_path,
                            f"{noise}-{noise_type}-{i}-results.txt.rank{self.dist.rank}",
                        )
                        writer[noise][noise_type][i] = open(res_file, "w")
                        noise_list.append(
                            os.path.join(
                                self.path.result_path,
                                f"{noise}-{noise_type}-{i}-results.txt.rank",
                            )
                        )
            noise_list = sorted(noise_list)
        else:
            res_file = os.path.join(
                self.path.result_path, f"results.txt.rank{self.dist.rank}"
            )
            writer = open(res_file, "w")

        for batch_idx, batch in enumerate(self.val_data["loader"]):
            if batch_idx % 10 == 0:
                info_str = f"[{batch_idx}/{len(self.val_data['loader'])}] "
                info_str += f"{batch_idx * 100 / len(self.val_data['loader']):.6f}%"
                self.logger.info(info_str)
            input = batch["image"]
            label = batch["label"]
            # Move data to the correct GPU for this rank
            device_id = self.dist.rank % torch.cuda.device_count()
            input = input.cuda(device=device_id)
            label = label.squeeze().view(-1).cuda(device=device_id).long()
            # compute output
            logits = self.model(input)
            scores = F.softmax(logits, dim=1)
            # compute prediction
            _, preds = logits.data.topk(k=1, dim=1)
            preds = preds.view(-1)
            # update batch information
            batch.update({"prediction": preds})
            batch.update({"score": scores})
            # save prediction information
            self.val_data["loader"].dataset.dump(writer, batch)
        if imagenetc_flag:
            for noise in writer:
                for noise_type in writer[noise]:
                    for i in range(1, 6):
                        writer[noise][noise_type][i].close()
        else:
            writer.close()
        link.barrier()
        if imagenetc_flag:
            for idx, file_prefix in enumerate(noise_list):
                if idx % self.dist.world_size == self.dist.rank:
                    # print(f"idx: {idx}, rank: {self.dist.rank}, {file_prefix}")
                    self.val_data["loader"].dataset.evaluate(file_prefix)
            link.barrier()
            if self.dist.rank == 0:
                self.val_data["loader"].dataset.merge_eval_res(self.path.result_path)
                self._write_imagenet_c_mce()
            metrics = {}
        else:
            if self.dist.rank == 0:
                metrics = self.val_data["loader"].dataset.evaluate(res_file)
                # Don't write acc_var_neg here - will be computed globally after all methods
                self.logger.info(json.dumps(metrics.metric, indent=2))
            else:
                metrics = {}
        link.barrier()

        # broadcast metrics to other process
        metrics = broadcast_object(metrics)
        # self.model.train()
        self.logger.info(f"{self.prefix_name} done.")
        return metrics

    def _write_acc_var_neg(self, res_file):
        """
        Compute variance of per-sample accuracy (0/1) and write its negative value.
        """
        prefix = res_file.rstrip("0123456789")
        merged_file = prefix.rsplit(".", 1)[0] + ".all"
        if not os.path.exists(merged_file):
            merged_file = self.val_data["loader"].dataset.merge(prefix)

        correct_list = []
        with open(merged_file, "r") as f:
            for line in f:
                try:
                    info = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "prediction" not in info or "label" not in info:
                    continue
                correct_list.append(1.0 if info["prediction"] == info["label"] else 0.0)

        if not correct_list:
            self.logger.warning("[acc-var] No valid predictions found.")
            return

        acc_mean = float(np.mean(correct_list))
        acc_var = float(np.var(correct_list))
        out_path = os.path.join(self.path.result_path, "acc_var_neg.json")
        with open(out_path, "w") as f:
            json.dump(
                {
                    "acc_mean": acc_mean,
                    "acc_var": acc_var,
                    "acc_var_neg": -acc_var,
                    "num_samples": len(correct_list),
                },
                f,
                indent=2,
            )
        self.logger.info(f"[acc-var] saved to: {out_path}")

    def _write_imagenet_c_mce(self):
        """
        Generate mCE summary JSON for ImageNet-C results if helper script exists.
        """
        script_path = os.path.join(self.path.root_path, "calculate_imagenet_c_mce.py")
        if not os.path.exists(script_path):
            self.logger.warning(
                f"[imagenet-c] mCE script not found: {script_path}"
            )
            return
        results_dir = os.path.join(self.path.root_path, self.prefix_name)
        output_path = os.path.join(self.path.result_path, "mce.json")
        try:
            subprocess.run(
                [sys.executable, script_path, results_dir, "--output", output_path],
                check=False,
            )
            self.logger.info(f"[imagenet-c] mCE saved to: {output_path}")
        except Exception as exc:
            self.logger.warning(f"[imagenet-c] mCE calculation failed: {exc}")


@link_dist
def main():
    parser = argparse.ArgumentParser(description="Classification Solver")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--evaluate", action="store_true")

    args = parser.parse_args()
    # build solver
    config = parse_config(args.config)
    status = open("status.txt", "a+")

    if hasattr(config, "eval_list") and config.get("eval_list") is not None:
        # Batch evaluation mode with eval_list
        if SPRING_MODELS_REGISTRY is None:
            raise ImportError(
                "SPRING_MODELS_REGISTRY is not available. Please install spring models or use standard model registry."
            )

        model_list_config = config["eval_list"]
        if isinstance(model_list_config, list):
            model_list = {"default": model_list_config}
            model_family = "default"
        elif isinstance(model_list_config, dict):
            model_list = model_list_config
            model_family = list(model_list.keys())[0] if model_list else "default"
        else:
            raise ValueError(f"Invalid eval_list format: {type(model_list_config)}")

        resize_mode = {
            "pil": ["bilinear", "nearest", "box", "hamming", "cubic", "lanczos"],
            "opencv": ["nearest", "bilinear", "area", "cubic", "lanczos"],
        }
        decoder_list = ["pil"]  # ["opencv", "kestrel"]
        for model_name in model_list[model_family]:
            acc_list = []  # Collect acc values from all methods
            for mode in resize_mode:
                for resize in resize_mode[mode]:
                    model = SPRING_MODELS_REGISTRY.get(model_name)(
                        pretrained=True,
                        num_classes=1000,
                        normalize={"type": "solo_bn"},
                        initializer={"method": "msra"},
                        frozen_layers=[],
                        task="classification",
                    )
                    tmp_config = deepcopy(config)
                    transforms = tmp_config["data"]["test"]["transforms"]
                    # Only update resize settings when transforms are list-based
                    if (
                        isinstance(transforms, list)
                        and transforms
                        and isinstance(transforms[0], dict)
                        and "kwargs" in transforms[0]
                    ):
                        transforms[0]["kwargs"]["mode"] = resize
                        transforms[0]["kwargs"]["backend"] = mode
                    else:
                        if link.get_rank() == 0:
                            print(
                                f"[multi_eval] Skip resize override for transforms type: {type(transforms)}"
                            )

                    solver = MultiEvalSolver_S(
                        tmp_config, model, f"{model_name}/{mode}.{resize}"
                    )
                    # evaluate or train
                    metrics = solver.evaluate()
                    # Collect acc value if available
                    if link.get_rank() == 0 and metrics and hasattr(metrics, "metric") and "top1" in metrics.metric:
                        acc_list.append(float(metrics.metric["top1"]))
                    status.write(f"{model_name}.{mode}.{resize} done\n")

            for decoder in decoder_list:
                model = SPRING_MODELS_REGISTRY.get(model_name)(
                    pretrained=True,
                    num_classes=1000,
                    normalize={"type": "solo_bn"},
                    initializer={"method": "msra"},
                    frozen_layers=[],
                    task="classification",
                )
                tmp_config = deepcopy(config)
                tmp_config["data"]["test"]["image_reader"]["type"] = decoder
                solver = MultiEvalSolver_S(tmp_config, model, f"{model_name}/{decoder}")
                # evaluate or train
                metrics = solver.evaluate()
                # Collect acc value if available
                if link.get_rank() == 0 and metrics and hasattr(metrics, "metric") and "top1" in metrics.metric:
                    acc_list.append(float(metrics.metric["top1"]))
                status.write(f"{model_name}.{decoder} done\n")

            # Compute global acc variance after all methods are evaluated
            if link.get_rank() == 0 and acc_list and config.data.test.get("save_acc_var_neg", False):
                acc_array = np.array(acc_list)
                acc_mean = float(np.mean(acc_array))
                acc_var = float(np.var(acc_array))
                # Save to model's main directory (e.g., clip_vit_l_14_fare2_clip/)
                model_result_path = os.path.join(os.getcwd(), model_name, "acc_var_neg.json")
                os.makedirs(os.path.dirname(model_result_path), exist_ok=True)
                with open(model_result_path, "w") as f:
                    json.dump(
                        {
                            "acc_mean": acc_mean,
                            "acc_var": acc_var,
                            "acc_var_neg": -acc_var,
                            "num_methods": len(acc_list),
                            "acc_list": acc_list,
                        },
                        f,
                        indent=2,
                    )
                print(f"[acc-var] Global acc variance saved to: {model_result_path}")
                status.write(f"[acc-var] Global acc variance: {acc_var}, neg: {-acc_var}\n")
    else:
        # Single model evaluation mode - load from saver config
        resize_mode = {
            "pil": ["bilinear", "nearest", "box", "hamming", "cubic", "lanczos"],
            "opencv": ["nearest", "bilinear", "area", "cubic", "lanczos"],
        }
        decoder_list = ["pil"]  # ["opencv", "kestrel"]

        # Test different resize modes
        acc_list = []  # Collect acc values from all methods
        for mode in resize_mode:
            for resize in resize_mode[mode]:
                tmp_config = deepcopy(config)
                transforms = tmp_config["data"]["test"]["transforms"]
                if (
                    isinstance(transforms, list)
                    and transforms
                    and isinstance(transforms[0], dict)
                    and "kwargs" in transforms[0]
                ):
                    transforms[0]["kwargs"]["mode"] = resize
                    transforms[0]["kwargs"]["backend"] = mode
                else:
                    if link.get_rank() == 0:
                        print(
                            f"[multi_eval] Skip resize override for transforms type: {type(transforms)}"
                        )

                solver = MultiEvalSolver_S(
                    tmp_config, None, f"{config.model.type}/{mode}.{resize}"
                )
                metrics = solver.evaluate()
                # Collect acc value if available
                if link.get_rank() == 0 and metrics and hasattr(metrics, "metric") and "top1" in metrics.metric:
                    acc_list.append(float(metrics.metric["top1"]))
                status.write(f"{config.model.type}/{mode}.{resize} done\n")

        # Test different decoders
        for decoder in decoder_list:
            tmp_config = deepcopy(config)
            tmp_config["data"]["test"]["image_reader"]["type"] = decoder
            solver = MultiEvalSolver_S(
                tmp_config, None, f"{config.model.type}/{decoder}"
            )
            metrics = solver.evaluate()
            # Collect acc value if available
            if link.get_rank() == 0 and metrics and hasattr(metrics, "metric") and "top1" in metrics.metric:
                acc_list.append(float(metrics.metric["top1"]))
            status.write(f"{config.model.type}/{decoder} done\n")
        
        # Compute global acc variance after all methods are evaluated
        if link.get_rank() == 0 and acc_list and config.data.test.get("save_acc_var_neg", False):
            acc_array = np.array(acc_list)
            acc_mean = float(np.mean(acc_array))
            acc_var = float(np.var(acc_array))
            # Save to model's main directory (e.g., clip_vit_l_14_fare2_clip/)
            model_result_path = os.path.join(os.getcwd(), config.model.type, "acc_var_neg.json")
            os.makedirs(os.path.dirname(model_result_path), exist_ok=True)
            with open(model_result_path, "w") as f:
                json.dump(
                    {
                        "acc_mean": acc_mean,
                        "acc_var": acc_var,
                        "acc_var_neg": -acc_var,
                        "num_methods": len(acc_list),
                        "acc_list": acc_list,
                    },
                    f,
                    indent=2,
                )
            print(f"[acc-var] Global acc variance saved to: {model_result_path}")
            status.write(f"[acc-var] Global acc variance: {acc_var}, neg: {-acc_var}\n")

    status.close()


if __name__ == "__main__":
    main()
