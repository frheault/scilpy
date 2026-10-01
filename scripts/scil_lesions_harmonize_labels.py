#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script harmonizes labels across a set of lesion files represented in
NIfTI format. It ensures that labels are consistent across multiple input
images by matching labels between images based on spatial proximity and
overlap criteria.

The script works iteratively, so the multiple inputs should be in chronological
order (and changing the order affects the output). All images should be
co-registered.

To obtain labels from binary mask use scil_labels_from_mask.py

WARNING: this script requires all files to have all lesions segmented.
If your data only show new lesions at each timepoints (common in manual
segmentation), use the option --incremental_lesions to merge past timepoints.
    T1 = T1, T2 = T1 + T2, T3 = T1 + T2 + T3
"""

import argparse
import os

import nibabel as nib
import numpy as np
import tqdm

from scilpy.image.labels import (get_data_as_labels,
                                 get_labels_from_mask)
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             assert_headers_compatible)
from scilpy.tractanalysis.reproducibility_measures import (
    compute_bundle_adjacency_voxel)


EPILOG = """
Reference:
    [1] Köhler, Caroline, et al. "Exploring individual multiple sclerosis
    lesion volume change over time: development of an algorithm for the
    analyses of longitudinal quantitative MRI measures."
    NeuroImage: Clinical 21 (2019): 101623.
"""


def harmonize_labels(original_data, min_voxel_overlap=1, max_adjacency=1e2):
    """
    Harmonize lesion labels across multiple 3D volumes by ensuring consistent
    labeling.

    This function takes multiple 3D NIfTI volumes with labeled regions
    (e.g., lesions) and harmonizes the labels so that regions that are the same
    across different volumes are assigned a consistent label. It operates by
    iteratively comparing labels in each volume to those in previous volumes
    and matching them based on spatial proximity and overlap.

    Parameters
    ----------
    original_data : list of numpy.ndarray
        A list of 3D numpy arrays where each array contains labeled regions.
        Labels should be non-zero integers, where each unique integer represents
        a different region or lesion.
    min_voxel_overlap : int, optional
        Minimum number of overlapping voxels required for two regions (lesions)
        from different volumes to be considered as potentially the same lesion.
        Default is 1.
    max_adjacency : float, optional
        Maximum distance allowed between the centroids of two regions for them
        to be considered as the same lesion. Default is 1e2 (infinite).

    Returns
    -------
    list of numpy.ndarray
        A list of 3D numpy arrays with the same shape as `original_data`, where
        labels have been harmonized across all volumes. Each region across
        volumes that is identified as the same will have the same label.
    """

    relabeled_data = [np.zeros_like(data) for data in original_data]
    relabeled_data[0] = original_data[0]
    labels = np.unique(original_data)[1:]

    # We will iterate over all possible combinations of labels
    N = len(original_data)
    total_iteration = ((N * (N - 1)) // 2)
    tqdm_bar = tqdm.tqdm(total=total_iteration, desc="Harmonizing labels")

    # We want to label images in order
    for first_pass in range(len(original_data)):
        unmatched_labels = np.unique(original_data[first_pass])[1:].tolist()
        best_match_score = {label: 999999 for label in labels}
        best_match_pos = {label: None for label in labels}

        # We iterate over all previous images to find the best match
        for second_pass in range(0, first_pass):
            tqdm_bar.update(1)

            # We check all existing labels in relabeled data
            for label_ind_1 in range(len(labels)):
                label_1 = labels[label_ind_1]

                if label_1 not in original_data[first_pass]:
                    continue

                # This check requires to at least overlap by N voxel
                coord_1 = np.where(original_data[first_pass] == label_1)
                overlap_labels_count = np.unique(relabeled_data[second_pass][coord_1],
                                                 return_counts=True)

                potential_labels_val = overlap_labels_count[0].tolist()
                potential_labels_count = overlap_labels_count[1].tolist()
                potential_labels = []
                for val, count in zip(potential_labels_val,
                                      potential_labels_count):
                    if val != 0 and count > min_voxel_overlap:
                        potential_labels.append(val)

                # We check all labels touching the previous label
                for label_2 in potential_labels:
                    tmp_data_1 = np.zeros_like(original_data[0])
                    tmp_data_2 = np.zeros_like(original_data[0])

                    # We always compare the previous relabeled data with the next
                    # original data
                    tmp_data_1[original_data[first_pass] == label_1] = 1
                    tmp_data_2[relabeled_data[second_pass] == label_2] = 1

                    # They should have a similar shape (TODO: parameters)
                    adjacency = compute_bundle_adjacency_voxel(
                        tmp_data_1, tmp_data_2)
                    if adjacency > max_adjacency:
                        continue

                    if adjacency < best_match_score[label_1]:
                        best_match_score[label_1] = adjacency
                        best_match_pos[label_1] = label_2

            # We relabel the data and keep track of the unmatched labels
            for label in labels:
                if best_match_pos[label] is not None:
                    old_label = label
                    new_label = best_match_pos[label]
                    relabeled_data[first_pass][original_data[first_pass]
                                               == old_label] = new_label
                    if old_label in unmatched_labels:
                        unmatched_labels.remove(old_label)

        # Anything that is left should be given a new label
        if first_pass == 0:
            continue
        next_label = np.max(relabeled_data[:first_pass]) + 1
        for label in unmatched_labels:
            relabeled_data[first_pass][original_data[first_pass]
                                       == label] = next_label
            next_label += 1

    return [data.astype(np.uint16) for data in relabeled_data]



def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_images', nargs='+',
                   help='Input file name, in nifti format.')
    p.add_argument('out_dir',
                   help='Output directory.')
    p.add_argument('--max_adjacency', type=float, default=5.0,
                   help='Maximum adjacency distance between lesions for '
                   'them to be considered as the potential match '
                   '[%(default)s].')
    p.add_argument('--min_voxel_overlap', type=int, default=1,
                   help='Minimum number of overlapping voxels between '
                   'lesions for them to be considered as the potential '
                   'match [%(default)s].')

    p.add_argument('--incremental_lesions', action='store_true',
                   help='If lesions files only show new lesions at each '
                        'timepoint, this will merge past timepoints.')
    p.add_argument('--debug_mode', action='store_true',
                   help='Add a fake voxel to the corner to ensure consistent '
                        'colors in MI-Brain.')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_images)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir)
    assert_headers_compatible(parser, args.in_images)

    imgs = [nib.load(filename) for filename in args.in_images]
    original_data = [get_data_as_labels(img) for img in imgs]

    masks = []
    if args.incremental_lesions:
        for i, data in enumerate(original_data):
            mask = np.zeros_like(data)
            mask[data > 0] = 1
            masks.append(mask)
            if i > 0:
                new_data = np.sum(masks, axis=0)
                new_data[new_data > 0] = 1
            else:
                new_data = mask
            original_data[i] = get_labels_from_mask(new_data)

    relabeled_data = harmonize_labels(original_data,
                                      args.min_voxel_overlap,
                                      max_adjacency=args.max_adjacency)

    max_label = np.max(relabeled_data) + 1
    for i, img in enumerate(imgs):
        if args.debug_mode:
            relabeled_data[i][0, 0, 0] = max_label  # To force identical color
        nib.save(nib.Nifti1Image(relabeled_data[i], img.affine),
                 os.path.join(args.out_dir, os.path.basename(args.in_images[i])))


if __name__ == "__main__":
    main()
