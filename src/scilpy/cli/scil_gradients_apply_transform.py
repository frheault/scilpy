#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform bvecs using an affine/rigid transformation.

"""

import argparse
import logging

import numpy as np

from scilpy.io.gradients import read_bvals_bvecs
from scilpy.io.stateful_gradient import StatefulGradient
from scilpy.io.stateful_image import StatefulImage
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_verbose_arg,
                             load_matrix_in_any_format)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_bvecs',
                   help='Path of the bvec file, in FSL format')
    p.add_argument('in_transfo',
                   help='Path of the file containing the 4x4 \n'
                        'transformation, matrix (.txt, .npy or .mat).')
    p.add_argument('out_bvecs',
                   help='Output filename of the transformed bvecs.')

    p.add_argument('--in_dwi',
                   help='Reference DWI image to handle affine-aware '
                        'bvec loading.\nIf not provided, identity RAS '
                        'is assumed.')

    p.add_argument('--inverse', action='store_true',
                   help='Apply the inverse transformation.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_bvecs, args.in_transfo])
    assert_outputs_exist(parser, args, args.out_bvecs)

    transfo = load_matrix_in_any_format(args.in_transfo)[:3, :3]

    if args.inverse:
        transfo = np.linalg.inv(transfo)

    if args.in_dwi:
        ref_simg = StatefulImage.load(args.in_dwi)
    else:
        # Identity assumption
        ref_simg = StatefulImage(np.zeros((1, 1, 1)), np.eye(4))
        ref_simg._original_affine = np.eye(4)

    sgrad = read_bvals_bvecs(None, args.in_bvecs, simg=ref_simg)

    # Apply transform to world-space vectors
    new_bvecs_ras = sgrad.to_ras() @ transfo

    # Save transformed bvecs using original affine of reference
    final_sgrad = StatefulGradient(np.zeros(len(new_bvecs_ras)),
                                   new_bvecs_ras, ref_simg, space='ras')
    final_sgrad.save('/tmp/dummy.bval', args.out_bvecs)


if __name__ == "__main__":
    main()
