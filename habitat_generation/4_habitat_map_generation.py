"""
Habitat Map Generation Script

This script generates habitat masks from clustering results by mapping superpixel clusters
back to the original image space. It creates spatial maps that visualize different habitat
regions within tumor volumes.

Key operations:
1. Loads clustering results (labels and optimal K)
2. Validates consistency between clustering labels and patient superpixel counts
3. Maps cluster labels to superpixel segments
4. Creates NIfTI habitat mask images preserving spatial information
5. Saves habitat masks to both central and patient-specific directories
6. Generates statistics on habitat distribution across patients
7. Creates visualization of habitat combination distribution

Dependencies:
- os
- json
- numpy
- pandas
- matplotlib
- SimpleITK
- collections

Usage:
Import this module and call the appropriate functions with parameters.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from collections import defaultdict


def create_habitat_mask(patient_dir, t1ce_path, seg_path, cluster_labels, best_k,
                        output_dir="Adaptive_KMeans_9_all600_mask"):
    try:
        sp_label = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        sp_label = np.transpose(sp_label, (1, 2, 0))
        habitat_mask = np.zeros_like(sp_label, dtype=np.int16)
        valid_sp_ids = np.unique(sp_label)[1:]

        if len(valid_sp_ids) != len(cluster_labels):
            return None

        for i, sp_id in enumerate(valid_sp_ids):
            habitat_mask[sp_label == sp_id] = cluster_labels[i] + 1

        original_img = sitk.ReadImage(t1ce_path)
        habitat_img = sitk.GetImageFromArray(np.transpose(habitat_mask, (2, 0, 1)))
        habitat_img.CopyInformation(original_img)

        patient_id = os.path.basename(patient_dir)
        os.makedirs(output_dir, exist_ok=True)
        patient_output_dir = os.path.join(output_dir, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)

        mask_filename = f"habitat_mask_centerA_k{best_k}.nii.gz"
        central_path = os.path.join(patient_output_dir, mask_filename)
        local_path = os.path.join(patient_dir, mask_filename)

        sitk.WriteImage(habitat_img, central_path)
        sitk.WriteImage(habitat_img, local_path)

        return (central_path, local_path, len(np.unique(habitat_mask[habitat_mask > 0])))

    except Exception:
        return None


def generate_habitat_maps(experiment_dir, t1ce_paths, seg_paths, patient_dirs,
                          output_excel=None, output_image=None):
    best_k_path = os.path.join(experiment_dir, "best_k.npy")
    best_k = np.load(best_k_path).item()

    labels_path = os.path.join(experiment_dir, f"labels_bestK_{best_k}.npy")
    if not os.path.exists(labels_path):
        labels_path = os.path.join(experiment_dir, "labels", f"labels_k{best_k}.npy")
    labels = np.load(labels_path)

    idx_count_path = os.path.join(experiment_dir, "idx_count.npy")
    idx_count = np.load(idx_count_path)

    if len(labels) != np.sum(idx_count):
        if len(labels) > np.sum(idx_count):
            labels = labels[:np.sum(idx_count)]

    valid_indices = []
    valid_t1ce_paths = []
    valid_seg_paths = []
    valid_dirs = []

    for i, (t1ce_path, seg_path, patient_dir) in enumerate(zip(t1ce_paths, seg_paths, patient_dirs)):
        if os.path.exists(t1ce_path) and os.path.exists(seg_path):
            valid_indices.append(i)
            valid_t1ce_paths.append(t1ce_path)
            valid_seg_paths.append(seg_path)
            valid_dirs.append(patient_dir)

    if len(valid_dirs) != len(idx_count):
        if len(valid_dirs) < len(idx_count):
            valid_labels = []
            valid_idx_count = []
            current_idx = 0
            for i in range(len(valid_dirs)):
                if i < len(idx_count):
                    count = idx_count[i]
                    start_idx = current_idx
                    end_idx = current_idx + count
                    if end_idx <= len(labels):
                        valid_labels.extend(labels[start_idx:end_idx])
                        valid_idx_count.append(count)
                    current_idx += count
            labels = np.array(valid_labels)
            idx_count = np.array(valid_idx_count)
        else:
            idx_count = idx_count[:len(valid_dirs)]

    label_patient_counter = defaultdict(int)
    combination_counter = defaultdict(int)
    patient_combinations = []

    current_idx = 0
    success_count = 0

    for i, (patient_dir, t1ce_path, seg_path) in enumerate(zip(valid_dirs, valid_t1ce_paths, valid_seg_paths)):
        if i < len(idx_count):
            patient_label_count = idx_count[i]
        else:
            patient_combinations.append("Invalid")
            continue

        if current_idx + patient_label_count <= len(labels):
            patient_labels = labels[current_idx:current_idx + patient_label_count]
        else:
            patient_combinations.append("Invalid")
            current_idx += patient_label_count
            continue

        current_idx += patient_label_count

        result = create_habitat_mask(
            patient_dir=patient_dir,
            t1ce_path=t1ce_path,
            seg_path=seg_path,
            cluster_labels=patient_labels,
            best_k=best_k,
            output_dir=f"Habitat_Masks_centerA_test_K{best_k}"
        )

        if result:
            unique_labels = set(np.unique(patient_labels))
            for label in unique_labels:
                label_patient_counter[label] += 1
            combo_key = frozenset(unique_labels)
            combination_counter[combo_key] += 1
            combo_str = '+'.join(map(str, sorted(unique_labels)))
            patient_combinations.append(combo_str)
            success_count += 1
        else:
            patient_combinations.append("Invalid")

    if output_excel:
        result_df = pd.DataFrame({
            'Patient_Index': range(len(patient_dirs)),
            'Directory': patient_dirs,
            'T1CE_Path': t1ce_paths,
            'Seg_Path': seg_paths,
            'Habitat_Combination': ['Invalid'] * len(patient_dirs)
        })

        for j, valid_idx in enumerate(valid_indices):
            if j < len(patient_combinations):
                result_df.loc[valid_idx, 'Habitat_Combination'] = patient_combinations[j]

        result_df.to_excel(output_excel, index=False)

    if output_image and combination_counter:
        sorted_combos = sorted(combination_counter.items(), key=lambda x: (len(x[0]), sorted(x[0])))
        combo_labels = ['+'.join(map(str, sorted(c))) for c, _ in sorted_combos]
        combo_counts = [count for _, count in sorted_combos]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(combo_labels, combo_counts, color='steelblue', edgecolor='black')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.ylabel("Patient Number", fontsize=12)
        plt.xlabel("Habitat Group", fontsize=12)
        plt.title(f"The distribution of habitat combinations in patients (K={best_k})", fontsize=14)

        for bar, count in zip(bars, combo_counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(count),
                     ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_image, dpi=300)
        plt.close()

    return {
        'success_count': success_count,
        'total_patients': len(valid_dirs),
        'label_distribution': dict(label_patient_counter),
        'combination_distribution': {
            '+'.join(map(str, sorted(k))): v for k, v in combination_counter.items()
        }
    }


if __name__ == "__main__":
    """
    Example Usage: Habitat Map Generation
    
    Input:
        experiment_dir: Directory with clustering results
        patient_dirs: List of patient directories
        t1ce_paths: List of T1CE image paths
        seg_paths: List of superpixel segmentation paths
    
    Output:
        habitat_mask_centerA_k{k}.nii.gz (per patient)
        habitat_results.xlsx
        habitat_distribution.png
    """
    experiment_dir = "./path/to/clustering"
    patient_dirs = ["./path/to/patient_001", "./path/to/patient_002"]
    t1ce_paths = ["./path/to/patient_001/t1ce.nii.gz", "./path/to/patient_002/t1ce.nii.gz"]
    seg_paths = ["./path/to/patient_001/superpixel_mask.nii.gz", "./path/to/patient_002/superpixel_mask.nii.gz"]
    output_excel = "./output/habitat_results.xlsx"
    output_image = "./output/habitat_distribution.png"
    