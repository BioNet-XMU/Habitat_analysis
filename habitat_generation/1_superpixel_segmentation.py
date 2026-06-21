"""
Superpixel Segmentation using Adaptive KMeans

Performs superpixel segmentation on multi-modal MRI images using KMeans clustering.

Input:
    image_paths: List of paths to 9 MRI modalities (NIfTI files)
                 Order: T1CE, T1W, T2W, T1CE_entropy, T1W_entropy, T2W_entropy,
                        T1CE_energy, T1W_energy, T2W_energy

Output:
    label_map: 3D numpy array of superpixel labels (saved as NIfTI file)

Dependencies: numpy, nibabel, sklearn.cluster.KMeans, pandas, os
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.cluster import KMeans


def load_images(image_paths):
    """Load multiple MRI modalities and stack into 4D array."""
    images = [nib.load(p).get_fdata() for p in image_paths]
    return np.stack(images, axis=-1)


def extract_features(image_data):
    """Extract features (9 imaging channels + XYZ coordinates)."""
    mask = np.any(image_data[..., :9] > 0, axis=-1)
    coords = np.where(mask)
    
    features = np.zeros((len(coords[0]), 12))
    
    # 9 imaging channels
    for i in range(9):
        features[:, i] = image_data[..., i][coords]
    
    # Normalized spatial coordinates
    for j, coord in enumerate(coords):
        features[:, 9 + j] = coord / image_data.shape[j]
    
    return features, coords, mask


def compute_adaptive_k(mask, voxel_per_cluster=500, k_min=30, k_max=300):
    """Compute adaptive number of clusters based on voxel count."""
    voxel_num = np.sum(mask)
    k = voxel_num // voxel_per_cluster
    return int(max(k_min, min(k, k_max)))


def perform_kmeans_segmentation(image_data):
    """Main function: perform KMeans clustering for superpixel segmentation."""
    features, coords, mask = extract_features(image_data)
    n_clusters = compute_adaptive_k(mask)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(features) + 1
    
    label_map = np.zeros(image_data.shape[:3], dtype=np.int32)
    label_map[coords] = labels
    
    return label_map, n_clusters


def process_patient(image_paths, patient_name, output_root):
    """Process a single patient: load images, segment, and save results."""
    try:
        # Load multi-modal image data
        image_data = load_images(image_paths)
        
        # Perform segmentation
        label_map, num_superpixels = perform_kmeans_segmentation(image_data)
        
        # Save output
        output_dir = os.path.join(output_root, patient_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as NIfTI (using first image as reference for affine)
        ref_img = nib.load(image_paths[0])
        output_img = nib.Nifti1Image(label_map.astype(np.int32), ref_img.affine)
        output_path = os.path.join(output_dir, f"{patient_name}_superpixel_seg.nii.gz")
        nib.save(output_img, output_path)
        
        return output_path, num_superpixels
    
    except Exception as e:
        print(f"Error processing {patient_name}: {str(e)}")
        return None, None


def main():
    """Main function for batch processing."""
    config = {
        'excel_path': './path/to/path.xlsx',
        'output_root': './output/superpixel_segmentation',
        'image_types': ['T1CE', 'T1W', 'T2W',
                        'T1CE_entropy', 'T1W_entropy', 'T2W_entropy',
                        'T1CE_energy', 'T1W_energy', 'T2W_energy']
    }
    
    os.makedirs(config['output_root'], exist_ok=True)
    
   
    
    for idx, row in df.iterrows():
        patient_name = row.get('PatientID', f'patient_{idx:03d}')
        image_paths = []
        valid = True
        
        for img_type in config['image_types']:
            path = row.get(img_type, None)
            if path and os.path.exists(path):
                image_paths.append(path)
            else:
                valid = False
                break
        
        if valid:
            process_patient(image_paths, patient_name, config['output_root'])


if __name__ == "__main__":
   main()