import cv2
import torch
import numpy as np
import pypose as pp
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
import yaml
from fdc.utils.rectify_video import VideoRectifier, RectificationConfig

from torch.utils.data import Dataset

from ..Interface import StereoFrame, StereoData
from ..SequenceBase import SequenceBase

from fdc.common.config import StereoDataConfig, SLAMConfig


class GeneralStereoSequence(SequenceBase[StereoFrame]):
    @classmethod
    def name(cls) -> str: return "GeneralStereo"
    
    def __init__(self, config: SimpleNamespace | dict[str, Any]) -> None:
        cfg = self.config_dict2ns(config)

        # metadata
        self.baseline = cfg.bl
        self.T_BS = pp.identity_SE3(1, dtype=torch.float64)

        # Check if using on-the-fly rectification
        data_config_file = cfg.data_config_file
        with open(data_config_file, 'r') as f:
            data_config = yaml.load(f, Loader=yaml.FullLoader)
        
        data_cfg = StereoDataConfig(**data_config['data'])
        slam_cfg = SLAMConfig(**data_config['slam'])
        self.ImageL = VideoRectificationDataset(data_cfg.rectify_config, "left")
        self.ImageR = VideoRectificationDataset(data_cfg.rectify_config, "right")

        assert len(self.ImageL) == len(self.ImageR)
        
        if hasattr(cfg.camera, "fx"):
            self.K = torch.tensor([[
                [cfg.camera.fx, 0., cfg.camera.cx], 
                [0., cfg.camera.fy, cfg.camera.cy],
                [0.           , 0., 1.           ]
            ]], dtype=torch.float).repeat(len(self.ImageL), 1, 1)
        else:
            raise ValueError("Camera intrinsic matrix is not provided")

        self.length = len(self.ImageL)
        super().__init__(self.length)

    def __getitem__(self, local_index: int) -> StereoFrame:
        index = self.get_index(local_index)
        imageL = self.ImageL[index]
        imageR = self.ImageR[index]
            
        return StereoFrame(
            idx    = [local_index],
            time_ns= [local_index * 1000],         # FIXME: a fake timestamp.
            stereo = StereoData(
                T_BS     = self.T_BS,
                K        = self.K[index:index+1],
                baseline = torch.tensor([self.baseline]),
                width    = imageL.size(-1),
                height   = imageL.size(-2),
                time_ns  = [local_index * 1000],   # FIXME: a fake timestamp.
                imageL   = imageL,
                imageR   = imageR
            )
        )

    @classmethod
    def is_valid_config(cls, config) -> None:
        cls._enforce_config_spec(config, {
            "root"  : lambda s: isinstance(s, str),
            "bl"    : lambda v: isinstance(v, float),
            "format": lambda s: isinstance(s, str),
            "camera": lambda v: isinstance(v, dict) and (len(v) == 0 or cls._enforce_config_spec(v, {
                "fx": lambda v: isinstance(v, float),
                "fy": lambda v: isinstance(v, float),
                "cx": lambda v: isinstance(v, float),
                "cy": lambda v: isinstance(v, float)
            }, allow_excessive_cfg=True)) or True
        })

class MonocularDataset(Dataset):
    """
    Return images in the given directory ends with .png
    Return the image in shape (1, 3, H, W) with dtype=float32 
    and normalized (image in [0, 1])
    """
    def __init__(self, directory: Path, format: Literal["png", "jpg"]) -> None:
        super().__init__()
        self.directory = directory
        assert self.directory.exists(), f"Monocular image directory {self.directory} does not exist"
        
        self.file_names = list(sorted(directory.glob(f"*.{format}")))
            
        self.length = len(self.file_names)
        assert self.length > 0, f"No file with '.png' suffix is found under {self.directory}"

    @staticmethod
    def load_png_format(path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(f"Failed to read image from {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        # Output image tensor in shape of (1, C, H, W)
        image = self.load_png_format(str(self.file_names[index]))
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        image /= 255.
        return image


class VideoRectificationDataset(Dataset):
    """
    Dataset that performs on-the-fly rectification from video.
    Returns images in the same format as MonocularDataset: (1, 3, H, W) float32 in [0, 1]
    """
    def __init__(self, rectify_cfg: RectificationConfig, side: Literal["left", "right"]) -> None:
        super().__init__()
        self.side = side
        # Create video rectifier
        self.rectifier = VideoRectifier(
            video_path=rectify_cfg.video_path,
            video_timestamps_path=rectify_cfg.video_timestamps_path,
            kalibr_yaml_path=rectify_cfg.kalibr_yaml_path,
            split_mode=rectify_cfg.split_mode,
            rotation=rectify_cfg.rotation,
            swap_left_right=rectify_cfg.swap_left_right,
            rectify_alpha=rectify_cfg.rectify_alpha,
            frame_range=rectify_cfg.frame_range
        )

        self.length = self.rectifier.get_frame_count()

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        # Rectify frame on-the-fly
        rect_left, rect_right = self.rectifier.read_and_rectify(index)

        if rect_left is None:
            raise RuntimeError(f"Failed to rectify frame {index}")

        # Select left or right
        image = rect_left if self.side == "left" else rect_right

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to tensor (1, 3, H, W) and normalize
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        image /= 255.
        return image

    def __del__(self):
        if hasattr(self, 'rectifier'):
            self.rectifier.release()