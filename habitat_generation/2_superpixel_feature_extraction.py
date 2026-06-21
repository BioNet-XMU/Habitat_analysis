"""
Superpixel Radiomic Feature Extraction Script

This script extracts radiomic features from superpixel segments using PyRadiomics.
Key features:
1. Safely loads NIfTI images using both SimpleITK and NiBabel (fallback)
2. Handles direction cosine issues in NIfTI files
3. Normalizes images to uint8 range for feature extraction
4. Expands single-voxel superpixels to 3x3x3 neighborhood when needed
5. Extracts first-order, GLCM, GLRLM, GLSZM, GLDM, NGTDM, and shape features
6. Tracks extraction success/failure statistics

Dependencies:
- numpy
- pandas
- SimpleITK
- nibabel
- pyradiomics
- warnings

Usage:
Import this module and call the process_patient function with appropriate parameters.
"""

import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor
import warnings

warnings.filterwarnings('ignore')


def load_with_nibabel(image_path):
    try:
        import nibabel as nib
        img = nib.load(image_path)
        data = img.get_fdata()
        if len(data.shape) == 4:
            data = data[:, :, :, 0]
        image_sitk = sitk.GetImageFromArray(data.astype(np.float32))
        try:
            zooms = img.header.get_zooms()
            if len(zooms) >= 3:
                spacing = [float(z) for z in zooms[:3]]
                image_sitk.SetSpacing(spacing)
        except Exception:
            image_sitk.SetSpacing((1.0, 1.0, 1.0))
        try:
            image_sitk.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        except Exception:
            pass
        return image_sitk
    except Exception:
        return None


def load_image_safely(image_path, use_nibabel=False):
    if use_nibabel:
        return load_with_nibabel(image_path)
    try:
        image = sitk.ReadImage(image_path)
        return image
    except Exception as e:
        if "orthonormal direction cosines" in str(e):
            return load_with_nibabel(image_path)
        else:
            return None


def normalize_image(image_array):
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    if max_val - min_val > 1e-8:
        normalized = (image_array - min_val) / (max_val - min_val)
        return (normalized * 255).astype(np.uint8)
    else:
        return np.zeros_like(image_array, dtype=np.uint8)


def get_unique_labels(mask_sitk):
    mask_array = sitk.GetArrayFromImage(mask_sitk)
    unique_labels = np.unique(mask_array)
    unique_labels = unique_labels[unique_labels != 0]
    return sorted(unique_labels)


def expand_single_pixel_roi(mask, label_num):
    try:
        mask_array = sitk.GetArrayFromImage(mask)
        mask_array = np.rint(mask_array).astype(int)
        indices = np.argwhere(mask_array == label_num)
        if indices.size == 0 or indices.size > 5:
            return None

        new_mask_array = np.zeros_like(mask_array, dtype=np.uint8)
        for idx in indices:
            z_idx, y_idx, x_idx = idx
            for dz in range(-1, 2):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        new_z = z_idx + dz
                        new_y = y_idx + dy
                        new_x = x_idx + dx
                        if (0 <= new_z < mask_array.shape[0] and
                                0 <= new_y < mask_array.shape[1] and
                                0 <= new_x < mask_array.shape[2]):
                            new_mask_array[new_z, new_y, new_x] = label_num

        expanded_voxels = np.sum(new_mask_array == label_num)
        if expanded_voxels > indices.size:
            expanded_mask = sitk.GetImageFromArray(new_mask_array)
            expanded_mask.CopyInformation(mask)
            return expanded_mask
        else:
            return None
    except Exception:
        return None


def extract_features_for_label(label_num, image, mask):
    mask_array = sitk.GetArrayFromImage(mask)
    roi_voxels = np.sum(mask_array == label_num)
    working_mask = mask

    if roi_voxels <= 5:
        expanded_mask = expand_single_pixel_roi(mask, label_num)
        if expanded_mask is not None:
            working_mask = expanded_mask

    settings = {
        'binWidth': 25,
        'label': label_num,
        'interpolator': sitk.sitkBSpline,
        'normalize': False,
        'normalizeScale': 100,
        'geometryTolerance': 1e6,
        'minimumROIDimensions': 1,
        'minimumROISize': 1,
        'correctMask': True,
        'force2D': False,
        'additionalInfo': True
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('gldm')
    extractor.enableFeatureClassByName('ngtdm')
    extractor.enableFeatureClassByName('shape')

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            featureVector = extractor.execute(image, working_mask)
        featureVector['Superpixel_Label'] = label_num
        featureVector['Original_Voxels'] = roi_voxels
        return (label_num, dict(sorted(featureVector.items())))
    except Exception:
        return (label_num, None)


def process_patient(image_path, mask_path, output_dir):
    patient_id = os.path.basename(os.path.dirname(image_path))
    os.makedirs(output_dir, exist_ok=True)

    image_sitk = load_image_safely(image_path)
    mask_sitk = load_image_safely(mask_path)

    if image_sitk is None or mask_sitk is None:
        return None

    image_size = image_sitk.GetSize()
    mask_size = mask_sitk.GetSize()

    if sorted(image_size) != sorted(mask_size):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image_sitk)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        mask_sitk = resampler.Execute(mask_sitk)
    elif image_size != mask_size:
        mask_array = sitk.GetArrayFromImage(mask_sitk)
        if image_size == (mask_size[2], mask_size[1], mask_size[0]):
            mask_array = np.transpose(mask_array, (2, 1, 0))
        mask_sitk = sitk.GetImageFromArray(mask_array)
        mask_sitk.CopyInformation(image_sitk)

    superpixel_labels = get_unique_labels(mask_sitk)

    if len(superpixel_labels) == 0:
        return None

    image_np = sitk.GetArrayFromImage(image_sitk)
    image_np = normalize_image(image_np)
    image_normalized = sitk.GetImageFromArray(image_np)
    image_normalized.CopyInformation(image_sitk)

    results = []
    for label in superpixel_labels:
        label_result = extract_features_for_label(label, image_normalized, mask_sitk)
        results.append(label_result)

    successful_results = [r[1] for r in results if r[1] is not None]

    if successful_results:
        df = pd.DataFrame(successful_results)
        metadata_cols_to_drop = [col for col in df.columns
                                 if col.startswith('diagnostics_') or
                                 col in ['PatientPath', 'version', 'imageHash', 'maskHash']]
        if metadata_cols_to_drop:
            df.drop(columns=metadata_cols_to_drop, inplace=True, errors='ignore')

        output_path = os.path.join(output_dir, f'{patient_id}_features.csv')
        df.to_csv(output_path, index=False)
        return output_path

    return None


if __name__ == "__main__":
    
    image_path = "./path/to/image.nii.gz"
    mask_path = "./path/to/superpixel_mask.nii.gz"
    output_dir = "./output/features"
    
    