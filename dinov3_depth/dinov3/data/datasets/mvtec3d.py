# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
import numpy as np
import torch

from .extended import ExtendedVisionDataset
from .decoders import Decoder, TargetDecoder


class DepthDecoder(Decoder):
    def __init__(self, data: bytes):
        self._data = data

    def decode(self) -> torch.Tensor:
        # self._data is the path encoded as bytes
        path = self._data.decode("utf-8")
        # # # arr = np.load(path, mmap_mode='r').astype(np.float32)  # (H, W)
        arr = np.load(path).astype(np.float32)  # (H, W)
        
        if arr.ndim == 2:
            arr = arr[:, :, None]  # add channel dim -> (H, W, 1)
        
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # -> (C, H, W)
        return tensor
    
class MVTec3D(ExtendedVisionDataset):
    def __init__(
        self,
        root: str,
        transforms=None,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=DepthDecoder,
            target_decoder=TargetDecoder,
        )
        self.files = [f for f in os.listdir(root) if f.endswith(".npy")]
        self.files.sort()

    def get_image_data(self, index: int) -> bytes:
        path = os.path.join(self.root, self.files[index])
        # return raw file path, so DepthDecoder can read it
        return path.encode("utf-8")

    def get_target(self, index: int):
        return None

    def __len__(self) -> int:
        return len(self.files)
