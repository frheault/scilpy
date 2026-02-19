#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script simply finds the 3 closest angular neighbors of each direction
(per shell) and compute the voxel-wise correlation.
If the angles or correlations to neighbors are below the shell average (by
args.std_scale x STD) it will flag the volume as a potential outlier.

This script supports multi-shells, but each shell is independant and detected
using the --b0_threshold parameter.

This script can be run before any processing to identify potential problem
before launching pre-processing.
"""

import argparse
import logging

from scilpy.dwi.operations import detect_volume_outliers
from scilpy.io.stateful_image import StatefulImage
from scilpy.io.utils import (add_b0_thresh_arg, add_skip_b0_check_arg,
                             add_stateful_gradient_args,
                             add_verbose_arg, assert_inputs_exist,
                             get_stateful_gradient_from_args)
from scilpy.gradients.bvec_bval_tools import check_b0_threshold
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_dwi',
                   help='The DWI file (.nii) to concatenate.')
    add_stateful_gradient_args(p, mandatory=True)

    p.add_argument('--std_scale', type=float, default=2.0,
                   help='How many deviation from the mean are required to be '
                        'considered an outlier. [%(default)s]')

    add_b0_thresh_arg(p)
    add_skip_b0_check_arg(p, will_overwrite_with_min=True)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose == "WARNING":
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec])

    vol = StatefulImage.load(args.in_dwi)
    data = vol.get_fdata()
    sgrad = get_stateful_gradient_from_args(args, vol)

    args.b0_threshold = check_b0_threshold(sgrad.bvals.min(),
                                           b0_thr=args.b0_threshold,
                                           skip_b0_check=args.skip_b0_check)

    # Not using the result. Only printing on screen. This is why the logging
    # level can never be set higher than INFO.
    detect_volume_outliers(data, sgrad.bvals, sgrad.bvecs, args.std_scale,
                           args.b0_threshold)


if __name__ == "__main__":
    main()
