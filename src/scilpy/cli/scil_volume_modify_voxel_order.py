#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modify the voxel order of an image.
Common voxel orders are RAS, LAS, LPS, etc.

This script will reorient the data and update the affine accordingly.
If bvals/bvecs are provided, the bvecs will be reoriented to match the new
axes system.
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.stateful_image import StatefulImage
from scilpy.io.stateful_gradient import StatefulGradient
from scilpy.io.utils import (add_overwrite_arg, add_stateful_gradient_args,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist, 
                             get_stateful_gradient_from_args)
from scilpy.utils.orientation import parse_voxel_order
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_image',
                   help='Path of the input volume.')
    p.add_argument('out_image',
                   help='Path of the output volume.')
    p.add_argument('--new_voxel_order',
                   help='New voxel order, e.g. RAS, LPS, etc.',
                   required=True)

    add_stateful_gradient_args(p, mandatory=False)
    p.add_argument('--out_bvec',
                   help='Path of the output bvec file.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_image, optional=[args.in_bval, args.in_bvec])
    assert_outputs_exist(parser, args, args.out_image, optional=args.out_bvec)

    if args.in_bvec and not args.out_bvec:
        parser.error('--out_bvec must be provided if --in_bvec is used.')

    simg = StatefulImage.load(args.in_image)

    parsed_voxel_order = parse_voxel_order(args.new_voxel_order,
                                           dimensions=len(simg.shape))

    # Reorient the in-memory data
    simg.reorient(parsed_voxel_order)

    # To ensure the new orientation is the one saved to disk,
    # we update the original orientation info.
    simg._original_axcodes = simg.axcodes
    simg._original_affine = simg.affine.copy()

    simg.save(args.out_image)

    if args.in_bvec:
        # Load gradients relative to the image
        sgrad = get_stateful_gradient_from_args(args, simg)
        
        # Save uses the current simg._original_affine (which we just updated to the new order)
        sgrad.save('/tmp/dummy.bval', args.out_bvec)


if __name__ == "__main__":
    main()
