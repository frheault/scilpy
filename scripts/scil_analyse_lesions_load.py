#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script will output informations about lesion load in bundle(s).
Either using as streamlines, binary map or a bundle voxel labels map.
"""

import argparse
import json
import os

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi


from scilpy.io.image import get_data_as_mask, get_data_as_label
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             add_json_args,
                             assert_output_dirs_exist_and_empty,
                             add_reference_arg)
from scilpy.segment.streamlines import filter_grid_roi
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.metrics_tools import compute_lesion_stats


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_lesion',
                   help='Binary mask of the lesion (.nii.gz).')
    p.add_argument('out_dir',
                   help='Output directory for lesion information')
    p1 = p.add_mutually_exclusive_group()
    p1.add_argument('--bundle',
                    help='Path of the bundle file (.trk).')
    p1.add_argument('--bundle_mask',
                    help='Path of the bundle binary mask.')
    p1.add_argument('--bundle_labels_map',
                    help='Path of the bundle labels_map.')

    p.add_argument('--min_lesion_vol', type=float, default=7,
                   help='Minimum lesion volume in mm3 [%(default)s].')
    p.add_argument('--out_atlas',
                   help='Save the lesion as an atlas.')
    p.add_argument('--out_lesion_stats', action='store_true',
                   help='Save the lesion-wise & streamlines count.')

    add_json_args(p)
    add_overwrite_arg(p)
    add_reference_arg(p)

    return p


def _save_json_wrapper(args, out_name, out_dict):
    filename = os.path.join(args.out_dir, out_name)
    with open(filename, 'w') as outfile:
        json.dump(out_dict, outfile,
                  sort_keys=args.sort_keys, indent=args.indent)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if (not args.bundle) and (not args.bundle_mask) \
            and (not args.bundle_labels_map):
        parser.error('One of the option --bundle or --map must be used')

    assert_inputs_exist(parser, [args.in_lesion],
                        optional=[args.bundle, args.bundle_mask,
                                  args.bundle_labels_map])
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir)

    lesion_img = nib.load(args.in_lesion)
    lesion_data = get_data_as_mask(lesion_img)

    if args.bundle:
        bundle_name, _ = split_name_with_nii(os.path.basename(args.bundle))
        sft = load_tractogram_with_reference(parser, args, args.bundle)
        sft.to_vox()
        sft.to_corner()
        streamlines = sft.get_streamlines_copy()
        map_data = compute_tract_counts_map(streamlines,
                                            lesion_data.shape)
        map_data[map_data > 0] = 1
    elif args.bundle_mask:
        bundle_name, _ = split_name_with_nii(
            os.path.basename(args.bundle_mask))
        map_img = nib.load(args.bundle_mask)
        map_data = get_data_as_mask(map_img)
    else:
        bundle_name, _ = split_name_with_nii(os.path.basename(
            args.bundle_labels_map))
        map_img = nib.load(args.bundle_labels_map)
        map_data = get_data_as_label(map_img)

    is_single_label = args.bundle_labels_map is None
    voxel_sizes = lesion_img.header.get_zooms()[0:3]
    lesion_atlas, _ = ndi.label(lesion_data)

    lesion_load_dict = compute_lesion_stats(
        map_data, lesion_atlas, single_label=is_single_label,
        voxel_sizes=voxel_sizes, min_lesion_vol=args.min_lesion_vol)

    if args.out_atlas:
        lesion_atlas *= map_data.astype(np.bool)
        nib.save(nib.Nifti1Image(lesion_atlas, lesion_img.affine),
                 args.out_atlas)

    if args.out_lesion_stats:
        lesion_dict = {}
        for lesion in np.unique(lesion_atlas)[1:]:
            curr_vol = np.count_nonzero(lesion_atlas[lesion_atlas == lesion]) \
                * np.prod(voxel_sizes)
            if curr_vol >= args.min_lesion_vol:
                key = str(lesion).zfill(3)
                lesion_dict[key] = {'volume': curr_vol}
                if args.bundle:
                    tmp = np.zeros(lesion_atlas.shape)
                    tmp[lesion_atlas == lesion] = 1
                    new_sft, _ = filter_grid_roi(sft, tmp, 'any', False)
                    lesion_dict[key]['strs_count'] = len(new_sft)

    if is_single_label:
        total_vol_dict = {}
        avg_volume_dict = {}
        std_volume_dict = {}
        lesion_count_dict = {}
        total_vol_dict[bundle_name] = lesion_load_dict['total_volume']
        avg_volume_dict[bundle_name] = lesion_load_dict['avg_volume']
        std_volume_dict[bundle_name] = lesion_load_dict['std_volume']
        lesion_count_dict[bundle_name] = lesion_load_dict['lesion_count']
    else:
        total_vol_dict = {bundle_name: {}}
        avg_volume_dict = {bundle_name: {}}
        std_volume_dict = {bundle_name: {}}
        lesion_count_dict = {bundle_name: {}}
        for key in lesion_load_dict.keys():
            total_vol_dict[bundle_name][key] = \
                lesion_load_dict[key]['total_volume']
            avg_volume_dict[bundle_name][key] = \
                lesion_load_dict[key]['avg_volume']
            std_volume_dict[bundle_name][key] = \
                lesion_load_dict[key]['std_volume']
            lesion_count_dict[bundle_name][key] = \
                lesion_load_dict[key]['lesion_count']

    _save_json_wrapper(args, 'sum_volume_per_label.json', total_vol_dict)
    _save_json_wrapper(args, 'avg_volume_per_label.json', avg_volume_dict)
    _save_json_wrapper(args, 'std_volume_per_label.json', std_volume_dict)
    _save_json_wrapper(args, 'lesion_count_per_label.json',
                       lesion_count_dict)

    if args.out_lesion_stats:
        vol_dict = {bundle_name: {}}
        streamlines_count_dict = {bundle_name: {}}
        for key in lesion_dict.keys():
            vol_dict[bundle_name][key] = lesion_dict[key]['volume']
            if args.bundle:
                streamlines_count_dict[bundle_name][key] = \
                    lesion_dict[key]['strs_count']

        _save_json_wrapper(args, 'volume_per_lesion.json', vol_dict)
        if args.bundle:
            _save_json_wrapper(args, 'streamlines_count_per_lesion.json',
                               streamlines_count_dict)


if __name__ == "__main__":
    main()