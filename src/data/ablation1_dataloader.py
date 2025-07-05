import autorootcwd
from monai.transforms import (
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandFlipd,
    CropForegroundd,
    Compose,
    Spacingd,
    AsDiscreted,
    GaussianSmoothd,
    Lambda,
)
from monai.data import CacheDataset, DataLoader, Dataset
import os
import SimpleITK as sitk
import torch
import lightning.pytorch as pl
from pathlib import Path

# 7 experiments
EXPERIMENT_CONFIGS = {
    1: {
        "name": "Original Data (clipping x, normalization x)",
        "clip_min": None,
        "clip_max": None,
        "normalize": False,
        "norm_min": -1,
        "norm_max": 1
    },
    2: {
        "name": "Normalization (clipping x, normalization -1~1)",
        "clip_min": None,
        "clip_max": None,
        "normalize": True,
        "norm_min": -1,
        "norm_max": 1
    },
    3: {
        "name": "Symmetric Clipping 1 (clipping -20~20, normalization -1~1)",
        "clip_min": -20.0,
        "clip_max": 20.0,
        "normalize": True,
        "norm_min": -1,
        "norm_max": 1
    },
    4: {
        "name": "Symmetric Clipping 2 (clipping -50~50, normalization -1~1)",
        "clip_min": -50.0,
        "clip_max": 50.0,
        "normalize": True,
        "norm_min": -1,
        "norm_max": 1
    },
    5: {
        "name": "Symmetric Clipping 3 (clipping -100~100, normalization -1~1)",
        "clip_min": -100.0,
        "clip_max": 100.0,
        "normalize": True,
        "norm_min": -1,
        "norm_max": 1
    },
    6: {
        "name": "Negative Clipping (clipping -50~0, normalization -1~1)",
        "clip_min": -50.0,
        "clip_max": 0.0,
        "normalize": True,
        "norm_min": -1,
        "norm_max": 1
    },
    7: {
        "name": "Positive Clipping (clipping 0~150, normalization 0~1)",
        "clip_min": 0.0,
        "clip_max": 150.0,
        "normalize": True,
        "norm_min": 0,
        "norm_max": 1
    },
}

def clip_normalize_distance_map(data, clip_min=None, clip_max=None, normalize=False, norm_min=-1, norm_max=1):
    """
    clip and normalize distance map
    """
    dist = data["seg"]  # [8, H, W, D]
    
    # clipping and normalization
    distance_maps = []
    
    for c in range(dist.shape[0]):  # each channel
        channel_dist = dist[c]  # [H, W, D]
        
        # clipping if clip_min and clip_max are provided
        if clip_min is not None and clip_max is not None:
            clipped_dist = torch.clamp(channel_dist, min=clip_min, max=clip_max)
        else:
            clipped_dist = channel_dist
        
        # normalize if normalize=True
        if normalize:
            if clip_min is not None and clip_max is not None:
                # use clipping range for normalization
                processed_dist = (clipped_dist - clip_min) / (clip_max - clip_min)
                processed_dist = processed_dist * (norm_max - norm_min) + norm_min
            else:
                # use actual data range for normalization
                data_min = clipped_dist.min()
                data_max = clipped_dist.max()
                processed_dist = (clipped_dist - data_min) / (data_max - data_min)
                processed_dist = processed_dist * (norm_max - norm_min) + norm_min
        else:
            processed_dist = clipped_dist
        
        distance_maps.append(processed_dist)
    
    processed_dist = torch.stack(distance_maps)

    
    data["seg"] = processed_dist
    return data

class CoronaryArteryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/imageCAS_ablation",
        batch_size: int = 4,
        patch_size: tuple = (96, 96, 96),
        num_workers: int = 4, 
        cache_rate: float = 0.05,
        experiment_id: int = 1
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.experiment_id = experiment_id
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        
        # setting experiment
        if experiment_id not in EXPERIMENT_CONFIGS:
            raise ValueError(f"Invalid experiment_id: {experiment_id}. Must be 1-7.")
        
        self.exp_config = EXPERIMENT_CONFIGS[experiment_id]
        
    def process_distance_map(self, data):
        
        return clip_normalize_distance_map(
            data,
            clip_min=self.exp_config["clip_min"],
            clip_max=self.exp_config["clip_max"],
            normalize=self.exp_config["normalize"],
            norm_min=self.exp_config["norm_min"],
            norm_max=self.exp_config["norm_max"]
        )
        
    def load_data_splits(self, split: str):
        split_dir = self.data_dir / split
        cases = sorted(os.listdir(split_dir))
        
        data_files = []
        for case in cases:
            case_dir = split_dir / case

            image_file = str(case_dir / "img.nii.gz")
            label_file = str(case_dir / "label.nii.gz")
            seg_file = str(case_dir / "distance_map.nii.gz")  # roi segmentation
            
            if os.path.exists(image_file) and os.path.exists(label_file):
                data_files.append({
                    "image": image_file,
                    "label": label_file,
                    "seg": seg_file
                })
        
        return data_files

    def prepare_data(self):
        transforms = [
            LoadImaged(keys=["image", "label", "seg"]),
            EnsureChannelFirstd(keys=["image", "label", "seg"]),
            Orientationd(keys=["image", "label", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-150,
                a_max=550,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "seg"], source_key="image"),
            Lambda(self.process_distance_map),
            RandCropByPosNegLabeld(
                keys=["image", "label", "seg"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label", "seg"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label", "seg"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandShiftIntensityd(keys="image", offsets=0.05, prob=0.5)
        ]

        self.train_transforms = Compose(transforms)

        val_transforms = [
            LoadImaged(keys=["image", "label", "seg"]),
            EnsureChannelFirstd(keys=["image", "label", "seg"]),
            Orientationd(keys=["image", "label", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-150,
                a_max=550,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "seg"], source_key="image"),
            Lambda(self.process_distance_map),
        ]

        self.val_transforms = Compose(val_transforms)

    def setup(self, stage=None):
        train_files = self.load_data_splits("train")
        val_files = self.load_data_splits("valid")
        test_files = self.load_data_splits("test")

        print(f"Found {len(train_files)} training cases")
        print(f"Found {len(val_files)} validation cases")
        print(f"Found {len(test_files)} test cases")

        # debug transforms
        print(f"Ablation 1: Experiment {self.experiment_id}: {self.exp_config['name']}")
        
        self.train_ds = CacheDataset(
            data=train_files,
            transform=self.train_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            copy_cache=False
        )

        self.val_ds = CacheDataset(
            data=val_files,
            transform=self.val_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            copy_cache=False
        )

        self.test_ds = CacheDataset(
            data=test_files,
            transform=self.val_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            copy_cache=False
        )

        # self.val_ds = Dataset(
        #     data=val_files,
        #     transform=self.val_transforms,
        # )

        # self.test_ds = Dataset(
        #     data=test_files,
        #     transform=self.val_transforms,
        # )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # pin_memory=True,
            # persistent_workers=True,
            # # prefetch_factor=1
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            num_workers=self.num_workers,
            # pin_memory=True,
            # persistent_workers=True,
            # # prefetch_factor=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            num_workers=self.num_workers,
            # pin_memory=True,
            # persistent_workers=False,
        )