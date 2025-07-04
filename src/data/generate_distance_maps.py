import os
import sys
import SimpleITK as sitk
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from monai.transforms import (
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    AsDiscreted,
    Compose,
)

def create_distance_map(binary_mask):
    """
    Create distance map from binary mask
    
    Args:
        binary_mask (torch.Tensor): [C, H, W, D] binary mask
        
    Returns:
        torch.Tensor: [C, H, W, D] distance map
    """
    distance_maps = []
    
    for c in range(binary_mask.shape[0]):  # each channel
        channel_mask = binary_mask[c].numpy()  # [H, W, D]
        
        # convert to SimpleITK image
        sitk_mask = sitk.GetImageFromArray(channel_mask)
        sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)
        
        # create distance map
        distance_map = sitk.SignedMaurerDistanceMap(
            sitk_mask,
            insideIsPositive=False,  # heart outside is positive
            squaredDistance=False,
            useImageSpacing=True     # physical distance (mm)
        )
        
        # convert to tensor
        distance_map_array = sitk.GetArrayFromImage(distance_map)
        distance_map_tensor = torch.from_numpy(distance_map_array)
        
        distance_maps.append(distance_map_tensor)
    
    return torch.stack(distance_maps)

def generate_distance_map_for_case(case_dir, output_dir):
    """
    Generate distance map for a single case
    
    Args:
        case_dir (Path): Path to the case directory
        output_dir (Path): Path to the output directory
    """
    seg_file = case_dir / "heart_combined.nii.gz"
    
    if not seg_file.exists():
        print(f"Warning: Segmentation file not found: {seg_file}")
        return False
    
    # load and transform segmentation
    transforms = Compose([
        LoadImaged(keys=["seg"]),
        EnsureChannelFirstd(keys=["seg"]),
        Orientationd(keys=["seg"], axcodes="RAS"),
        AsDiscreted(keys=["seg"], to_onehot=8),
    ])
    
    data = {"seg": str(seg_file)}
    transformed_data = transforms(data)
    seg = transformed_data["seg"]  # [C, H, W, D] one-hot encoded
    
    # create distance map
    distance_map = create_distance_map(seg)
    
    # save distance map as nifti file
    case_name = case_dir.name
    output_case_dir = output_dir / case_name
    output_case_dir.mkdir(parents=True, exist_ok=True)
    
    # save distance map
    distance_map_file = output_case_dir / "distance_map.nii.gz"
    
    # convert tensor to numpy and save as nifti
    distance_map_np = distance_map.numpy()  # [C, H, W, D]
    
    # load original seg file to get metadata
    original_seg = sitk.ReadImage(str(seg_file))
    
    # For 4D image: [C, H, W, D] -> [C, D, W, H] (SimpleITK format)
    # SimpleITK will reverse the order: [C, D, W, H] -> [H, W, D, C] when loaded
    distance_map_np = np.transpose(distance_map_np, (0, 3, 2, 1))  # [C, D, W, H]
    
    sitk_distance_map = sitk.GetImageFromArray(distance_map_np, isVector=False)
    
    # Copy spacing, origin, direction from original 3D image
    # For 4D image, add spacing for the 4th dimension (channel dimension)
    original_spacing = original_seg.GetSpacing()
    spacing_4d = original_spacing + (1.0,)  # Add 1.0 for channel dimension
    sitk_distance_map.SetSpacing(spacing_4d)
    
    original_origin = original_seg.GetOrigin()
    origin_4d = original_origin + (0.0,)  # Add 0.0 for channel dimension
    sitk_distance_map.SetOrigin(origin_4d)
    
    # Direction matrix for 4D: extend 3x3 to 4x4
    original_direction = original_seg.GetDirection()
    # Convert 3x3 to 4x4 matrix: add 0s to each row and identity for 4th dimension
    direction_4d = [
        original_direction[0], original_direction[1], original_direction[2], 0.0,  # row 1
        original_direction[3], original_direction[4], original_direction[5], 0.0,  # row 2  
        original_direction[6], original_direction[7], original_direction[8], 0.0,  # row 3
        0.0, 0.0, 0.0, 1.0  # row 4
    ]
    sitk_distance_map.SetDirection(direction_4d)
    
    sitk.WriteImage(sitk_distance_map, str(distance_map_file))
    
    print(f"Generated distance map for case {case_name}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate distance maps from segmentation files")
    parser.add_argument("--data_dir", type=str, default="data/imageCAS_ablation", 
                        help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="data/imageCAS_ablation_distance", 
                        help="Directory to save the distance maps")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"], 
                        help="Data splits to process")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # copy data_splits.yaml if it exists
    if (data_dir / "data_splits.yaml").exists():
        import shutil
        shutil.copy2(data_dir / "data_splits.yaml", output_dir / "data_splits.yaml")
    
    total_cases = 0
    successful_cases = 0
    
    for split in args.splits:
        split_dir = data_dir / split
        output_split_dir = output_dir / split
        
        if not split_dir.exists():
            print(f"Warning: Split directory {split_dir} does not exist")
            continue
        
        output_split_dir.mkdir(parents=True, exist_ok=True)
        
        # get all cases in this split
        cases = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        print(f"Processing {len(cases)} cases in {split} split...")
        
        for case_dir in tqdm(cases, desc=f"Processing {split}"):
            total_cases += 1
            if generate_distance_map_for_case(case_dir, output_split_dir):
                successful_cases += 1
    
    print(f"\nCompleted processing {successful_cases}/{total_cases} cases")
    print(f"Distance maps saved to: {output_dir}")

if __name__ == "__main__":
    main()