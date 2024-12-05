import nrrd
import numpy as np
from ultralytics import YOLO
import cv2
import torch
from tqdm import tqdm
import SimpleITK as sitk
import matplotlib.pyplot as plt


def normalize_volume(volume_data):
    volume_min = np.nanmin(volume_data[~np.isnan(volume_data)])
    volume_max = np.nanmax(volume_data[~np.isnan(volume_data)])
    normalized = np.zeros_like(volume_data)
    mask = ~np.isnan(volume_data)
    normalized[mask] = (
        (volume_data[mask] - volume_min) * 255 / (volume_max - volume_min)
    )
    return normalized.astype(np.uint8)


def apply_3d_level_set(
    volume,
    initial_mask,
    num_iterations=100,
    propagation_scaling=1.0,
    curvature_scaling=1.0,
    advection_scaling=1.0,
):
    """
    Apply ITK's 3D geodesic active contour level set to refine segmentation
    """
    itk_volume = sitk.GetImageFromArray(volume.astype(np.float32))
    itk_mask = sitk.GetImageFromArray((initial_mask > 127).astype(np.float32))

    spacing = itk_volume.GetSpacing()
    itk_volume.SetSpacing(spacing)
    itk_mask.SetSpacing(spacing)

    gradient = sitk.GradientMagnitudeRecursiveGaussian(itk_volume, sigma=1.0)
    sigmoid = sitk.Sigmoid(gradient, alpha=2.0, beta=2.0)

    level_set = sitk.GeodesicActiveContourLevelSetImageFilter()
    level_set.SetPropagationScaling(propagation_scaling)
    level_set.SetCurvatureScaling(curvature_scaling)
    level_set.SetAdvectionScaling(advection_scaling)
    level_set.SetMaximumRMSError(0.02)
    level_set.SetNumberOfIterations(num_iterations)

    refined_mask = level_set.Execute(itk_mask, sigmoid)
    refined_mask_array = sitk.GetArrayFromImage(refined_mask)
    return (refined_mask_array > 0.0).astype(np.uint8) * 255


def apply_3d_morphological_operations(binary_volume):
    """
    Apply 3D morphological operations using SimpleITK
    """
    itk_volume = sitk.GetImageFromArray(binary_volume)
    closing = sitk.BinaryMorphologicalClosing(itk_volume > 127, kernelRadius=[2, 2, 2])
    opening = sitk.BinaryMorphologicalOpening(closing, kernelRadius=[1, 1, 1])
    return sitk.GetArrayFromImage(opening).astype(np.uint8) * 255


def process_3d_with_multiple_outputs(
    input_path,
    yolo_output_path,
    morph_output_path,
    refined_output_path,
    model_path="best.pt",
):
    """
    Process 3D volume and generate three different outputs:
    1. Raw YOLO predictions
    2. YOLO + morphological operations only
    3. Full pipeline with level set refinement
    """
    print("Loading data...")
    data, header = nrrd.read(input_path)

    print("Loading model...")
    model = YOLO(model_path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # First pass: Get YOLO predictions
    print("Performing initial YOLO segmentation...")
    sagittal_data = np.transpose(data, (2, 0, 1))
    normalized_volume = normalize_volume(sagittal_data)
    initial_masks = np.zeros_like(sagittal_data)

    total_slices = sagittal_data.shape[0]
    start_slice = int(total_slices * 0.1)
    end_slice = int(total_slices * 0.9)
    original_size = normalized_volume[0].shape

    for i in tqdm(range(start_slice, end_slice), desc="YOLO segmentation"):
        slice_norm = normalized_volume[i]
        slice_resized = cv2.resize(slice_norm, (640, 640))
        slice_rgb = cv2.cvtColor(slice_resized, cv2.COLOR_GRAY2RGB)

        results = model(slice_rgb, device=device)
        if results and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask, original_size)
            initial_masks[i] = mask_resized

    # Save YOLO output (Output 1)
    print("Saving YOLO segmentation...")
    yolo_masks = np.transpose(initial_masks, (1, 2, 0))
    nrrd.write(yolo_output_path, yolo_masks, header)

    # Apply morphological operations only (Output 2)
    print("Applying morphological operations...")
    morph_only = apply_3d_morphological_operations(yolo_masks)
    nrrd.write(morph_output_path, morph_only, header)

    # Convert to original orientation for level set processing
    normalized_volume = np.transpose(normalized_volume, (1, 2, 0))

    # Apply full pipeline with level set refinement (Output 3)
    print("Applying level set refinement...")
    refined_segmentation = apply_3d_level_set(
        normalized_volume,
        yolo_masks,
        num_iterations=200,
        propagation_scaling=1.0,
        curvature_scaling=1.0,
        advection_scaling=1.0,
    )

    # Final morphological cleanup on refined segmentation
    print("Performing final cleanup...")
    final_refined = apply_3d_morphological_operations(refined_segmentation)
    nrrd.write(refined_output_path, final_refined, header)

    return yolo_masks, morph_only, final_refined


# Example usage
if __name__ == "__main__":
    yolo_output, morph_output, refined_output = process_3d_with_multiple_outputs(
        "MRBrainTumor1.nrrd",
        "tumor_segmentation_yolo.nrrd",
        "tumor_segmentation_morph.nrrd",
        "tumor_segmentation_refined.nrrd",
    )