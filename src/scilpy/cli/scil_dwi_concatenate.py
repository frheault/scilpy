#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concatenate DWI, bval and bvecs together. File must be specified in matching
order. Default data type will be the same as the first input DWI.

"""

import argparse
import logging

from dipy.io.utils import is_header_compatible
import nibabel as nib
import numpy as np

from scilpy.io.gradients import read_bvals_bvecs
from scilpy.io.stateful_image import StatefulImage
from scilpy.io.stateful_gradient import StatefulGradient
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('out_dwi',
                   help='The name of the output DWI file.')
    p.add_argument('out_bval',
                   help='The name of the output b-values file (.bval).')
    p.add_argument('out_bvec',
                   help='The name of the output b-vectors file (.bvec).')

    p.add_argument('--in_dwis', nargs='+',
                   help='The DWI file (.nii) to concatenate.')
    p.add_argument('--in_bvals', nargs='+',
                   help='The b-values files in FSL format (.bval).')
    p.add_argument('--in_bvecs', nargs='+',
                   help='The b-vectors files in FSL format (.bvec).')

    p.add_argument('--data_type',
                   help='Data type of the output image. Use the format: '
                        'uint8, int16, int/float32, int/float64.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    if len(args.in_dwis) != len(args.in_bvals) \
            or len(args.in_dwis) != len(args.in_bvecs):
        parser.error('DWI, bvals and bvecs must have the same length')

    assert_inputs_exist(parser, args.in_dwis + args.in_bvals + args.in_bvecs)
    assert_outputs_exist(parser, args, [args.out_dwi, args.out_bval,
                                        args.out_bvec])

    all_bvals = []
    all_bvecs_rasmm = []
    total_size = 0
    ref_dwi = StatefulImage.load(args.in_dwis[0])

    # Process first input
    sgrad = read_bvals_bvecs(args.in_bvals[0], args.in_bvecs[0], simg=ref_dwi)
    total_size += len(sgrad.bvals)
    all_bvals.append(sgrad.bvals)
    all_bvecs_rasmm.append(sgrad.to_rasmm())

    all_dwi = np.zeros(ref_dwi.shape[0:3] + (0,), dtype=args.data_type)
    # We will build all_dwi list and concatenate at once for better efficiency
    # if it was many small ones, but here we follow the original logic of pre-allocating
    # or just concatenating data.
    # Actually, the original script pre-allocates based on total_size.
    # I need total_size first.

    for i in range(1, len(args.in_dwis)):
        curr_dwi = StatefulImage.load(args.in_dwis[i])
        if not is_header_compatible(curr_dwi, ref_dwi):
            raise ValueError('All DWI must have the compatible header.')

        curr_sgrad = read_bvals_bvecs(args.in_bvals[i], args.in_bvecs[i],
                                      simg=curr_dwi)
        if len(curr_sgrad.bvals) != curr_dwi.shape[-1]:
            raise ValueError('Paired bvals and DWI must have the same size.')

        total_size += len(curr_sgrad.bvals)
        all_bvals.append(curr_sgrad.bvals)
        all_bvecs_rasmm.append(curr_sgrad.to_rasmm())

    all_bvals = np.concatenate(all_bvals)
    all_bvecs_rasmm = np.concatenate(all_bvecs_rasmm)

    all_dwi = np.zeros(ref_dwi.shape[0:3] + (total_size,),
                       dtype=args.data_type or ref_dwi.get_data_dtype())

    last_count = 0
    for i in range(len(args.in_dwis)):
        curr_dwi = StatefulImage.load(args.in_dwis[i])
        curr_size = curr_dwi.shape[-1]
        all_dwi[..., last_count:last_count + curr_size] = curr_dwi.get_fdata()
        last_count += curr_size

    # Save results
    # Create final StatefulGradient to save correctly
    final_sgrad = StatefulGradient(all_bvals, all_bvecs_rasmm, ref_dwi, space='rasmm')
    final_sgrad.save(args.out_bval, args.out_bvec)

    res_img = nib.Nifti1Image(all_dwi, ref_dwi.affine, header=ref_dwi.header)
    StatefulImage.create_from(res_img, ref_dwi).save(args.out_dwi)


if __name__ == "__main__":
    main()
