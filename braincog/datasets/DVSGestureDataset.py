import tonic
import torch
import numpy as np
import torchvision.transforms as tvt
from typing import Tuple
from torch.utils.data import Dataset
from tonic.transforms import ToFrame

class DVSGestureDataset(Dataset):
    """
    将 DVSGesture 的事件帧序列汇聚为单张图，转为 3 通道，交给 CLIP 的 preprocess。
    """
    def __init__(
        self,
        root: str,
        train: bool,
        transform = None,
        target_transform = None,
        download: bool = False,
        time_window: int = 100_000,
        reduce: str = "sum",   # "sum" 或 "mean"
    ):
        self.train = train
        self.base = tonic.datasets.DVSGesture(
            save_to=root,
            transform=ToFrame(
                sensor_size=tonic.datasets.DVSGesture.sensor_size,
                time_window=time_window
            ),
            train=self.train
        )
        self.targets = self.base.targets  # tonic 提供的标签列表
        self.reduce = reduce
        self.to_pil = tvt.ToPILImage()

    def _frames_to_image(self, frames):
        """
        frames 形状可能是：
        - torch.Tensor [T, C, H, W] （常见，C=2 为极性）
        - numpy.ndarray 同形
        - 偶见 [T, H, W, C]
        这里做鲁棒处理，然后在 (T, C) 维度聚合为 [H, W]。
        """
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)

        # 统一到 [T, C, H, W]
        if frames.ndim == 4:
            if frames.shape[-1] in (1, 2):  # [T, H, W, C]
                frames = frames.permute(0, 3, 1, 2)  # -> [T, C, H, W]
            # 否则认为已经是 [T, C, H, W]
        else:
            raise ValueError(f"Unexpected frames shape: {frames.shape}")

        # 聚合：在 T、C 两个维度上 sum / mean -> [H, W]
        if self.reduce == "sum":
            img = frames.sum(dim=(0, 1))
        elif self.reduce == "mean":
            img = frames.float().mean(dim=(0, 1))
        else:
            raise ValueError(f"Unknown reduce: {self.reduce}")

        # 归一化到 [0,1]
        img = img.float()
        img = img - img.min()
        denom = img.max().clamp_min(1e-6)
        img = img / denom  # [H, W], float32

        # 变为 3 通道以适配 CLIP（ToPILImage 要求 CxHxW 或 HxW）
        img3 = img.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]

        # 转 PIL 再走 CLIP 的 preprocess
        pil_img = self.to_pil(img3)
        return pil_img

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        frames, label = self.base[idx]  # frames: [T, C, H, W] or [T, H, W, C]
        pil_img = self._frames_to_image(frames)

        img = tvt.Compose([
            tvt.Resize((64, 64)),
            tvt.ToTensor(),
            tvt.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                            std=(0.26862954, 0.26130258, 0.27577711))
        ])(pil_img)

        return img, int(label)

    def __len__(self):
        return len(self.base)