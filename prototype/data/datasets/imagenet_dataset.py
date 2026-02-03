import os.path as osp
import os
import json
import hashlib
import requests
import time
import numpy as np
import glob

from .base_dataset import BaseDataset
from prototype.prototype.data.image_reader import build_image_reader

import prototype.spring.linklink as link
import random
import torch

_FILENAME_INDEX_CACHE = {}


class ImageNetDataset(BaseDataset):
    """
    ImageNet Dataset.

    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - read_type (:obj:`str`): read type from the original meta_file
        - evaluator (:obj:`Evaluator`): evaluate to get metrics
        - image_reader_type (:obj:`str`): reader type 'pil' or 'ks'
        - server_cfg (list): server configurations

    Metafile example::
        "n01440764/n01440764_10026.JPEG 0\n"
    """

    def __init__(
        self,
        root_dir,
        meta_file,
        transform=None,
        read_from="mc",
        evaluator=None,
        image_reader_type="pil",
        server_cfg={},
    ):

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.read_from = read_from
        self.transform = transform
        self.evaluator = evaluator
        self.image_reader = build_image_reader(image_reader_type)
        self.initialized = False
        self.use_server = False
        self._file_path_cache = {}  # Cache for found file paths

        if len(server_cfg) == 0:
            # read from local file
            with open(meta_file) as f:
                lines = f.readlines()

            self.num = len(lines)
            self.metas = []
            for line in lines:
                filename, label = line.rstrip().split()
                self.metas.append({"filename": filename, "label": label})
        else:
            # read from http server
            self.server_ip = server_cfg["ip"]
            self.server_port = server_cfg["port"]
            self.use_server = True
            if isinstance(self.server_ip, str):
                self.server_ip = [self.server_ip]
            if isinstance(self.server_port, int):
                self.server_port = [self.server_port]
            assert len(self.server_ip) == len(
                self.server_port
            ), "length of ips should equal to the length of ports"

            self.num = int(
                requests.get(
                    "http://{}:{}/get_len".format(
                        self.server_ip[0], self.server_port[0]
                    )
                ).json()
            )

        super(ImageNetDataset, self).__init__(
            root_dir=root_dir,
            meta_file=meta_file,
            read_from=read_from,
            transform=transform,
            evaluator=evaluator,
        )
        # Optional filename index for faster lookup when meta entries
        # don't include subdirectories.
        self._filename_index = None
        self._filename_index_ready = False
        self._maybe_build_filename_index()

    def __len__(self):
        return self.num

    def _index_cache_key(self):
        try:
            meta_mtime = osp.getmtime(self.meta_file)
        except (OSError, FileNotFoundError):
            meta_mtime = None
        return (self.root_dir, self.meta_file, meta_mtime)

    def _index_cache_path(self):
        try:
            meta_mtime = osp.getmtime(self.meta_file)
        except (OSError, FileNotFoundError):
            meta_mtime = None
        cache_id = f"{self.meta_file}:{meta_mtime}"
        digest = hashlib.md5(cache_id.encode("utf-8")).hexdigest()[:12]
        return osp.join(self.root_dir, f".filename_index_{digest}.json")

    def _maybe_build_filename_index(self):
        if self._filename_index_ready:
            return
        self._filename_index_ready = True
        if self.use_server:
            return
        if not self.metas:
            return
        if not osp.isdir(self.root_dir):
            return
        sample_name = self.metas[0]["filename"]
        if "/" in sample_name or "\\" in sample_name:
            return

        cache_key = self._index_cache_key()
        cached = _FILENAME_INDEX_CACHE.get(cache_key)
        if cached:
            self._filename_index = cached
            return

        cache_path = self._index_cache_path()
        if osp.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cached_obj = json.load(f)
                if (
                    cached_obj.get("meta_file") == self.meta_file
                    and cached_obj.get("index")
                ):
                    self._filename_index = cached_obj["index"]
                    _FILENAME_INDEX_CACHE[cache_key] = self._filename_index
                    return
            except Exception:
                pass

        targets = {meta["filename"] for meta in self.metas}
        index = {}
        for root, _, files in os.walk(self.root_dir):
            for fname in files:
                if fname in targets and fname not in index:
                    index[fname] = osp.join(root, fname)
                    if len(index) == len(targets):
                        self._filename_index = index
                        _FILENAME_INDEX_CACHE[cache_key] = self._filename_index
                        try:
                            with open(cache_path, "w") as f:
                                json.dump(
                                    {
                                        "meta_file": self.meta_file,
                                        "index": self._filename_index,
                                    },
                                    f,
                                )
                        except Exception:
                            pass
                        return
        self._filename_index = index
        _FILENAME_INDEX_CACHE[cache_key] = self._filename_index
        try:
            with open(cache_path, "w") as f:
                json.dump({"meta_file": self.meta_file, "index": self._filename_index}, f)
        except Exception:
            pass

    def _find_file_in_subdirs(self, root_dir, filename):
        """
        Find a file in subdirectories if it's not found directly.
        This handles ImageNet validation data organized in class folders.
        """
        if self._filename_index is not None and filename in self._filename_index:
            return self._filename_index[filename]
        # Check cache first
        if filename in self._file_path_cache:
            return self._file_path_cache[filename]

        # First try the direct path
        direct_path = osp.join(root_dir, filename)
        if osp.exists(direct_path) and osp.isfile(direct_path):
            self._file_path_cache[filename] = direct_path
            return direct_path

        # If filename doesn't contain a path separator, search in subdirectories
        if "/" not in filename and "\\" not in filename:
            # Search for the file in all subdirectories
            pattern = osp.join(root_dir, "**", filename)
            matches = glob.glob(pattern, recursive=True)
            if matches:
                found_path = matches[0]
                self._file_path_cache[filename] = found_path
                return found_path

        # If still not found, return the original path (will raise error later)
        return direct_path

    def _load_meta(self, idx):
        if self.use_server:
            while True:
                # random select a server ip
                rdx = np.random.randint(len(self.server_ip))
                r_ip, r_port = self.server_ip[rdx], self.server_port[rdx]
                # require meta information
                try:
                    meta = requests.get(
                        "http://{}:{}/get/{}".format(r_ip, r_port, idx), timeout=500
                    ).json()
                    break
                except Exception:
                    time.sleep(0.005)

            return meta
        else:
            return self.metas[idx]

    def __getitem__(self, idx):
        curr_meta = self._load_meta(idx)
        # Find the actual file path (may be in subdirectories)
        filename = self._find_file_in_subdirs(self.root_dir, curr_meta["filename"])
        label = int(curr_meta["label"])
        # add root_dir to filename
        curr_meta["filename"] = filename
        img_bytes = self.read_file(curr_meta)
        img = self.image_reader(img_bytes, filename)

        if self.transform is not None:
            img = self.transform(img)

        item = {"image": img, "image_id": idx, "label": label, "filename": filename}
        return item

    def dump(self, writer, output):
        prediction = self.tensor2numpy(output["prediction"])
        label = self.tensor2numpy(output["label"])
        score = self.tensor2numpy(output["score"])

        if "filename" in output:
            # pytorch type: {'image', 'label', 'filename', 'image_id'}
            filename = output["filename"]
            image_id = output["image_id"]
            for _idx in range(prediction.shape[0]):
                res = {
                    "filename": filename[_idx],
                    "image_id": int(image_id[_idx]),
                    "prediction": int(prediction[_idx]),
                    "label": int(label[_idx]),
                    "score": [float("%.8f" % s) for s in score[_idx]],
                }
                writer.write(json.dumps(res, ensure_ascii=False) + "\n")
        else:
            # dali type: {'image', 'label'}
            for _idx in range(prediction.shape[0]):
                res = {
                    "prediction": int(prediction[_idx]),
                    "label": int(label[_idx]),
                    "score": [float("%.8f" % s) for s in score[_idx]],
                }
                writer.write(json.dumps(res, ensure_ascii=False) + "\n")
        writer.flush()


class RankedImageNetDataset(BaseDataset):
    def __init__(
        self,
        root_dir,
        meta_file,
        transform=None,
        read_from="mc",
        evaluator=None,
        image_reader_type="pil",
        server_cfg={},
    ):
        self.rank = link.get_rank()
        self.world_size = link.get_world_size()
        self.root_dir = root_dir
        self.meta_file = meta_file
        self.read_from = read_from
        self.transform = transform
        self.evaluator = evaluator
        self.image_reader = build_image_reader(image_reader_type)
        self.initialized = False
        self.use_server = False
        self._file_path_cache = {}  # Cache for found file paths

        if len(server_cfg) == 0:
            # read from local file
            ranked_meta = [[] for _ in range(self.world_size)]
            with open(meta_file) as f:
                random.seed(0)
                np.random.seed(0)
                for line in f:
                    filename, label = line.rstrip().split()
                    random_rank = random.randint(0, self.world_size - 1)
                    if self.rank == random_rank:
                        ranked_meta[random_rank].append(
                            {"filename": filename, "label": label}
                        )
            self.metas = ranked_meta[self.rank]
            self.num = len(self.metas)

            # balance data length in each subprocess
            ranked_num = [torch.LongTensor([0]) for _ in range(self.world_size)]
            link.allgather(ranked_num, torch.LongTensor([self.num]))
            link.barrier()
            max_num = max([item.item() for item in ranked_num])
            if max_num > self.num:
                diff = max_num - self.num
                self.metas.extend(random.sample(self.metas, diff))
                self.num = len(self.metas)
            else:
                assert self.num == max_num
        else:
            raise NotImplementedError

        super(RankedImageNetDataset, self).__init__(
            root_dir=root_dir,
            meta_file=meta_file,
            read_from=read_from,
            transform=transform,
            evaluator=evaluator,
        )
        # Optional filename index for faster lookup when meta entries
        # don't include subdirectories.
        self._filename_index = None
        self._filename_index_ready = False
        self._maybe_build_filename_index()

    def __len__(self):
        return self.num

    def _index_cache_key(self):
        try:
            meta_mtime = osp.getmtime(self.meta_file)
        except (OSError, FileNotFoundError):
            meta_mtime = None
        return (self.root_dir, self.meta_file, meta_mtime, self.rank)

    def _index_cache_path(self):
        try:
            meta_mtime = osp.getmtime(self.meta_file)
        except (OSError, FileNotFoundError):
            meta_mtime = None
        cache_id = f"{self.meta_file}:{meta_mtime}:rank{self.rank}"
        digest = hashlib.md5(cache_id.encode("utf-8")).hexdigest()[:12]
        return osp.join(self.root_dir, f".filename_index_{digest}.json")

    def _maybe_build_filename_index(self):
        if self._filename_index_ready:
            return
        self._filename_index_ready = True
        if self.use_server:
            return
        if not self.metas:
            return
        if not osp.isdir(self.root_dir):
            return
        sample_name = self.metas[0]["filename"]
        if "/" in sample_name or "\\" in sample_name:
            return

        cache_key = self._index_cache_key()
        cached = _FILENAME_INDEX_CACHE.get(cache_key)
        if cached:
            self._filename_index = cached
            return

        cache_path = self._index_cache_path()
        if osp.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cached_obj = json.load(f)
                if (
                    cached_obj.get("meta_file") == self.meta_file
                    and cached_obj.get("index")
                ):
                    self._filename_index = cached_obj["index"]
                    _FILENAME_INDEX_CACHE[cache_key] = self._filename_index
                    return
            except Exception:
                pass

        targets = {meta["filename"] for meta in self.metas}
        index = {}
        for root, _, files in os.walk(self.root_dir):
            for fname in files:
                if fname in targets and fname not in index:
                    index[fname] = osp.join(root, fname)
                    if len(index) == len(targets):
                        self._filename_index = index
                        _FILENAME_INDEX_CACHE[cache_key] = self._filename_index
                        try:
                            with open(cache_path, "w") as f:
                                json.dump(
                                    {
                                        "meta_file": self.meta_file,
                                        "index": self._filename_index,
                                    },
                                    f,
                                )
                        except Exception:
                            pass
                        return
        self._filename_index = index
        _FILENAME_INDEX_CACHE[cache_key] = self._filename_index
        try:
            with open(cache_path, "w") as f:
                json.dump({"meta_file": self.meta_file, "index": self._filename_index}, f)
        except Exception:
            pass

    def _find_file_in_subdirs(self, root_dir, filename):
        """
        Find a file in subdirectories if it's not found directly.
        This handles ImageNet validation data organized in class folders.
        """
        if self._filename_index is not None and filename in self._filename_index:
            return self._filename_index[filename]
        # Check cache first
        if filename in self._file_path_cache:
            return self._file_path_cache[filename]

        # First try the direct path
        direct_path = osp.join(root_dir, filename)
        if osp.exists(direct_path) and osp.isfile(direct_path):
            self._file_path_cache[filename] = direct_path
            return direct_path

        # If filename doesn't contain a path separator, search in subdirectories
        if "/" not in filename and "\\" not in filename:
            # Search for the file in all subdirectories
            pattern = osp.join(root_dir, "**", filename)
            matches = glob.glob(pattern, recursive=True)
            if matches:
                found_path = matches[0]
                self._file_path_cache[filename] = found_path
                return found_path

        # If still not found, return the original path (will raise error later)
        return direct_path

    def _load_meta(self, idx):
        if self.use_server:
            raise NotImplementedError
        else:
            return self.metas[idx]

    def __getitem__(self, idx):
        curr_meta = self._load_meta(idx)
        # Find the actual file path (may be in subdirectories)
        filename = self._find_file_in_subdirs(self.root_dir, curr_meta["filename"])
        label = int(curr_meta["label"])
        # add root_dir to filename
        curr_meta["filename"] = filename
        img_bytes = self.read_file(curr_meta)
        img = self.image_reader(img_bytes, filename)

        if self.transform is not None:
            img = self.transform(img)

        item = {"image": img, "label": label, "image_id": idx, "filename": filename}
        return item

    def dump(self, writer, output):
        prediction = self.tensor2numpy(output["prediction"])
        label = self.tensor2numpy(output["label"])
        score = self.tensor2numpy(output["score"])

        if "filename" in output:
            # pytorch type: {'image', 'label', 'filename', 'image_id'}
            filename = output["filename"]
            image_id = output["image_id"]
            for _idx in range(prediction.shape[0]):
                res = {
                    "filename": filename[_idx],
                    "image_id": int(image_id[_idx]),
                    "prediction": int(prediction[_idx]),
                    "label": int(label[_idx]),
                    "score": [float("%.8f" % s) for s in score[_idx]],
                }
                writer.write(json.dumps(res, ensure_ascii=False) + "\n")
        else:
            # dali type: {'image', 'label'}
            for _idx in range(prediction.shape[0]):
                res = {
                    "prediction": int(prediction[_idx]),
                    "label": int(label[_idx]),
                    "score": [float("%.8f" % s) for s in score[_idx]],
                }
                writer.write(json.dumps(res, ensure_ascii=False) + "\n")
        writer.flush()
