#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to obtain labels from a binary mask which contains multiple blobs.

The script will assign a label to each blob in the mask. The background label
is excluded from the labels list. By default, the background label is 0 and
the labels are assigned in increasing order starting from 1.
"""


import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.image.labels import get_labels_from_mask
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             add_verbose_arg, assert_outputs_exist)
from scilpy.version import version_string

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def get_labels_from_mask(mask_data, labels=None, background_label=0,
                         min_voxel_count=0, min_distance=None):
    """
    Get labels from a binary mask which contains multiple blobs. Each blob
    will be assigned a label, by default starting from 1. Background will
    be assigned the background_label value.

    Parameters
    ----------
    mask_data: np.ndarray
        The mask data.
    labels: list, optional
        Labels to assign to each blobs in the mask. Excludes the background
        label.
    background_label: int
        Label for the background.
    min_voxel_count: int, optional
        Minimum number of voxels for a blob to be considered. Blobs with fewer
        voxels will be ignored.
    min_distance : int, optional
        If set, triggers a watershed algorithm to separate confluent blobs.
        The algorithm computes the Euclidean distance transform of the mask
        and finds local maxima (blob centers). This parameter is the minimum
        number of voxels separating two maxima for them to be considered
        distinct blobs. Smaller values split more aggressively; larger values
        merge nearby detections. If None, no watershed is performed and
        connected components are used directly.

    Returns
    -------
    label_map: np.ndarray
        The labels.
    """
    # Get the number of structures and assign labels to each blob
    if min_distance is not None:
        distance = ndi.distance_transform_edt(mask_data)
        coords = peak_local_max(
            distance,
            min_distance=min_distance,
            labels=mask_data,
            threshold_abs=0
        )
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        label_map = watershed(-distance, markers, mask=mask_data)
        nb_structures = np.max(label_map)
    else:
        # Any contiguous regions touching even by a single voxel will be
        # considered a single label.
        label_map, nb_structures = ndi.label(mask_data)

    if min_voxel_count:
        new_count = 0
        for label in range(1, nb_structures + 1):
            if np.count_nonzero(label_map == label) < min_voxel_count:
                label_map[label_map == label] = 0
            else:
                new_count += 1
                label_map[label_map == label] = new_count
        logging.debug(
            f"Ignored blob {nb_structures-new_count} with fewer "
            "than {min_voxel_count} voxels")
        nb_structures = new_count

    # Assign labels to each blob if provided
    if labels:
        # Only keep the first nb_structures labels if the number of labels
        # provided is greater than the number of blobs in the mask.
        if len(labels) > nb_structures:
            logging.warning("Number of labels ({}) does not match the number "
                            "of blobs in the mask ({}). Only the first {} "
                            "labels will be used.".format(
                                len(labels), nb_structures, nb_structures))
        # Cannot assign fewer labels than the number of blobs in the mask.
        elif len(labels) < nb_structures:
            raise ValueError("Number of labels ({}) is less than the number of"
                             " blobs in the mask ({}).".format(
                                 len(labels), nb_structures))

        # Copy the label map to avoid scenarios where the label list contains
        # labels that are already present in the label map
        custom_label_map = label_map.copy()
        # Assign labels to each blob
        for idx, label in enumerate(labels[:nb_structures]):
            custom_label_map[label_map == idx + 1] = label
        label_map = custom_label_map

    logging.info('Assigned labels {} to the mask.'.format(
        np.unique(label_map[label_map != background_label])))

    if background_label != 0 and background_label in label_map:
        logging.warning("Background label {} corresponds to a label "
                        "already in the map. This will cause issues.".format(
                            background_label))

    # Assign background label
    if background_label:
        label_map[label_map == 0] = background_label

    return label_map

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_mask', type=str, help='Input mask file.')
    p.add_argument('out_labels', type=str, help='Output label file.')

    p.add_argument('--labels', nargs='+', default=None, type=int,
                   help='Labels to assign to each blobs in the mask. '
                        'Excludes the background label.')
    p.add_argument('--background_label', default=0, type=int,
                   help='Label to assign to the background. [%(default)s]')

    p.add_argument('--min_volume', type=float, default=7,
                   help='Minimum volume in mm3 [%(default)s],'
                        'Useful for lesions.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_mask)
    assert_outputs_exist(parser, args, args.out_labels)
    # Load mask and get data
    mask_img = nib.load(args.in_mask)
    mask_data = get_data_as_mask(mask_img)
    voxel_volume = np.prod(np.diag(mask_img.affine)[:3])
    min_voxel_count = args.min_volume // voxel_volume

    # Get labels from mask
    label_map = get_labels_from_mask(
        mask_data, args.labels, args.background_label,
        min_voxel_count=min_voxel_count)
    # Save result
    out_img = nib.Nifti1Image(label_map.astype(np.uint16), mask_img.affine)
    nib.save(out_img, args.out_labels)


if __name__ == "__main__":
    main()
