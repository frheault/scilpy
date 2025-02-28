#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import argparse
import logging

from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from dipy.tracking.streamlinespeed import compress_streamlines
import nibabel as nib
import numpy as np

from scilpy.image.labels import get_data_as_labels
from scilpy.io.image import get_data_as_mask
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             add_reference_arg,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist,
                             add_compression_arg)
from scilpy.tractograms.streamline_and_mask_operations import \
    cut_streamlines_with_mask, cut_streamlines_between_labels, \
    CuttingStyle
from scilpy.tractograms.streamline_operations import \
    cut_invalid_streamlines, filter_streamlines_by_length

# Mapping the arguments to the cutting style
# (keep_longest, trim_endpoints) -> CuttingStyle
args_to_style = {(False, False): CuttingStyle.DEFAULT,
                 (True, False): CuttingStyle.KEEP_LONGEST}


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_tractogram',
                   help='Input tractogram file.')
    p.add_argument('target_reference',
                     help='Reference file (that was cropped) to get the '
                            'boundaries of the cropping.')
    p.add_argument('out_tractogram',
                     help='Output tractogram file.')
    p.add_argument('--resample', dest='step_size', type=float, default=None,
                   help='Resample streamlines to a specific step-size in mm '
                        '[%(default)s].')
    p.add_argument('--min_length', type=float, default=0,
                   help='Minimum length of streamlines to keep (in mm) '
                        '[%(default)s].')

    p.add_argument('--keep_longest', action='store_true',
                    help='If set, will keep the longest segment of the '
                         'streamline that is within the mask.')

    add_compression_arg(p)
    add_overwrite_arg(p)
    add_processes_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_tractogram, args.target_reference])
    assert_outputs_exist(parser, args, args.out_tractogram)

    # Loading
    ori_sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    tgt_img = nib.load(args.target_reference)

    sft = StatefulTractogram(ori_sft.streamlines, tgt_img, Space.RASMM,
                             origin=Origin.NIFTI)

    if len(sft.streamlines) == 0:
        parser.error('Input tractogram is empty.')

    new_sft, _ = cut_invalid_streamlines(sft)
    new_sft, _ = filter_streamlines_by_length(new_sft, args.min_length)
    new_sft.to_vox()
    new_sft.to_corner()

    # Saving
    if len(new_sft) == 0:
        logging.warning('No streamline intersected the provided mask. '
                        'Saving empty tractogram.')

    save_tractogram(new_sft, args.out_tractogram)


if __name__ == "__main__":
    main()
