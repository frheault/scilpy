#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script computes two myelin indices maps from the Magnetization Transfer
(MT) images.
Magnetization Transfer is a contrast mechanism in tissue resulting from the
proton exchange between non-aqueous protons (from macromolecules and their
closely associated water molecules, the "bound" pool) and protons in the free
water pool called aqueous protons. This exchange attenuates the MRI signal,
introducing microstructure-dependent contrast. MT's effect reflects the
relative density of macromolecules such as proteins and lipids, it has been
associated with myelin content in white matter of the brain.

Different contrasts can be done with an off-resonance pulse to saturating the
protons on non-aqueous molecules a frequency irradiation. The MT maps are
obtained using three or four contrasts: a single positive frequency image
and/or a single negative frequency image, and two unsaturated contrasts as
reference. These two references should be acquired with predominant PD
(proton density) and T1 weighting at different excitation flip angles
(a_PD, a_T1) and repetition times (TR_PD, TR_T1).


Input Data recommendation:
  - it is recommended to use dcm2niix (v1.0.20200331) to convert data
    https://github.com/rordenlab/dcm2niix/releases/tag/v1.0.20200331
  - dcm2niix conversion will create all echo files for each contrast and
    corresponding json files
  - all input must have a matching json file with the same filename
  - all contrasts must have a same number of echoes and coregistered
    between them before running the script.
  - Mask must be coregistered to the echo images
  - ANTs can be used for the registration steps (http://stnava.github.io/ANTs/)


The output consists of a MT_native_maps folder containing the 2 myelin maps:
    - MTR.nii.gz : Magnetization Transfer Ratio map
    The MT ratio is a measure reflecting the amount of bound protons.
    - MTsat.nii.gz : Magnetization Transfer saturation map
    The MT saturation is a pseudo-quantitative maps representing
    the signal change between the bound and free water pools.

As an option, the Complementary_maps folder contains the following images:
    - positive.nii.gz : single positive frequency image
    - negative.nii.gz : single negative frequency image
    - mtoff_PD.nii.gz : unsaturated proton density weighted image
    - mtoff_T1.nii.gz : unsaturated T1 weighted image
    - MTsat_sp.nii.gz : MTsat computed from the single positive frequency image
    - MTsat_sn.nii.gz : MTsat computed from the single negative frequency image
    - R1app.nii.gz : Apparent R1 map computed for MTsat.
    - B1_map.nii.gz : B1 map after correction and smoothing (if given).


The final maps from MT_native_maps can be corrected for B1+ field
  inhomogeneity, using either an empiric method with
  --in_B1_map option, suffix *B1_corrected is added for each map.
  --B1_correction_method empiric
  or a model-based method with
  --in_B1_map option, suffix *B1_corrected is added for each map.
  --B1_correction_method model_based
  --B1_fitValues 1 or 2 .mat files, obtained externally from
    https://github.com/TardifLab/OptimizeIHMTimaging/tree/master/b1Correction,
    and given in this order: positive frequency saturation, negative frequency
    saturation.
For both methods, the nominal value of the B1 map can be set with
  --B1_nominal value


>>> scil_mti_maps_MT.py path/to/output/directory
    --in_mtoff_pd path/to/echo*mtoff.nii.gz
    --in_positive path/to/echo*pos.nii.gz --in_negative path/to/echo*neg.nii.gz
    --in_mtoff_t1 path/to/echo*T1w.nii.gz --mask path/to/mask_bin.nii.gz
    --in_jsons path/to/echo*mtoff.json path/to/echo*T1w.json

By default, the script uses all the echoes available in the input folder.
If you want to use a single echo, replace the * with the specific number of
the echo.
"""

import argparse
import logging
import os
import sys

import nibabel as nib
import numpy as np

from scilpy.io.utils import (get_acq_parameters, add_overwrite_arg,
                             assert_inputs_exist, add_verbose_arg,
                             assert_output_dirs_exist_and_empty)
from scilpy.io.image import load_img
from scilpy.image.volume_math import concatenate
from scilpy.reconst.mti import (adjust_B1_map_intensities,
                                apply_B1_corr_empiric,
                                apply_B1_corr_model_based,
                                compute_ratio_map,
                                compute_saturation_map,
                                process_contrast_map,
                                threshold_map,
                                smooth_B1_map)

EPILOG = """
Helms G, Dathe H, Kallenberg K, Dechent P. High-resolution maps of
magnetization transfer with inherent correction for RF inhomogeneity
and T1 relaxation obtained from 3D FLASH MRI. Magnetic Resonance in Medicine.
2008;60(6):1396-407.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('out_dir',
                   help='Path to output folder.')
    p.add_argument('--out_prefix',
                   help='Prefix to be used for each output image.')
    p.add_argument('--mask',
                   help='Path to the binary brain mask.')
    p.add_argument('--extended', action='store_true',
                   help='If set, outputs the folder Complementary_maps.')
    p.add_argument('--filtering', action='store_true',
                   help='Gaussian filtering to remove Gibbs ringing. '
                        'Not recommended.')

    g = p.add_argument_group(title='Contrast maps', description='Path to '
                             'echoes corresponding to contrast images. All '
                             'constrasts must have \nthe same number of '
                             'echoes and coregistered between them. '
                             'Use * to include all echoes. \nThe in_mtoff_pd '
                             'input and at least one of in_positive or '
                             'in_negative are required.')
    g.add_argument("--in_positive", nargs='+',
                   required='--in_negative' not in sys.argv,
                   help='Path to all echoes corresponding to the '
                        'positive frequency \nsaturation pulse.')
    g.add_argument("--in_negative", nargs='+',
                   required='--in_positive' not in sys.argv,
                   help='Path to all echoes corresponding to the '
                        'negative frequency \nsaturation pulse.')
    g.add_argument("--in_mtoff_pd", nargs='+', required=True,
                   help='Path to all echoes corresponding to the predominant '
                        'PD \n(proton density) weighting images with no '
                        'saturation pulse.')
    g.add_argument("--in_mtoff_t1", nargs='+',
                   help='Path to all echoes corresponding to the predominant '
                        'T1 \nweighting images with no saturation pulse. This '
                        'one is optional, \nsince it is only needed for the '
                        'calculation of MTsat. \nAcquisition '
                        'parameters should also be set with this image.')

    a = p.add_argument_group(title='Acquisition parameters',
                             description='Acquisition parameters required '
                                         'for MTsat and ihMTsat '
                                         'calculation. \nThese are the '
                                         'excitation flip angles '
                                         '(a_PD, a_T1), in DEGREES, and \n'
                                         'repetition times (TR_PD, TR_T1) of '
                                         'the PD and T1 images, in SECONDS. '
                                         '\nCan be given through json files '
                                         '(--in_jsons) or directly '
                                         '(--in_acq_parameters).')
    a1 = a.add_mutually_exclusive_group(required='--in_mtoff_t1' in sys.argv)
    a1.add_argument('--in_jsons', nargs=2,
                    metavar=('PD_json', 'T1_json'),
                    help='Path to MToff PD json file and MToff T1 json file, '
                         'in that order. \nThe acquisition parameters will be '
                         'extracted from these files. \nMust come from a '
                         'Philips acquisition, otherwise, use '
                         'in_acq_parameters.')
    a1.add_argument('--in_acq_parameters', nargs=4, type=float,
                    metavar=('PD flip angle', 'T1 flip angle',
                             'PD repetition time', 'T1 repetition time'),
                    help='Acquisition parameters in that order: flip angle of '
                         'mtoff_PD, \nflip angle of mtoff_T1, repetition time '
                         'of mtoff_PD, \nrepetition time of mtoff_T1')

    b = p.add_argument_group(title='B1 correction')
    b.add_argument('--in_B1_map',
                   help='Path to B1 coregister map to MT contrasts.')
    b.add_argument('--B1_correction_method',
                   choices=['empiric', 'model_based'], default='empiric',
                   help='Choice of B1 correction method. Choose between '
                        'empiric and model-based. \nNote that the model-based '
                        'method requires a B1 fitvalues file. \nBoth method '
                        'will only correct the saturation measures. '
                        '[%(default)s]')
    b.add_argument('--B1_fitvalues', nargs='+',
                   help='Path to B1 fitvalues files obtained externally. '
                        'Should be one .mat \nfile per input MT-on image, '
                        'given in this specific order: \npositive frequency '
                        'saturation, negative frequency saturation.')
    b.add_argument('--B1_nominal', default=100, type=float,
                   help='Nominal value for the B1 map. For Philips, should be '
                        '100. [%(default)s]')
    b.add_argument('--B1_smooth_dims', default=5, type=int,
                   help='Dimension of the squared window used for B1 '
                        'smoothing, in number of voxels. [%(default)s]')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    output_dir = os.path.join(args.out_dir, 'MT_native_maps')
    if args.extended:
        extended_dir = os.path.join(args.out_dir, 'Complementary_maps')
        assert_output_dirs_exist_and_empty(parser, args, extended_dir,
                                           output_dir, create_dir=True)
    else:
        assert_output_dirs_exist_and_empty(parser, args, output_dir,
                                           create_dir=True)

    # Merge all echos path into a list
    input_maps = []
    contrast_names = []
    if args.in_positive:
        input_maps.append(args.in_positive)
        contrast_names.append('positive')
    if args.in_negative:
        input_maps.append(args.in_negative)
        contrast_names.append('negative')
    input_maps.append(args.in_mtoff_pd)
    contrast_names.append('mtoff_PD')
    if args.in_mtoff_t1:
        input_maps.append(args.in_mtoff_t1)
        contrast_names.append('mtoff_T1')

    # check data
    assert_inputs_exist(parser, args.in_mtoff_pd)  # Problem with maps_flat...
    # cannot verify the not required input. Somehow it breaks the input_maps...
    # even if it is not linked at all. WTF.
    for curr_map in input_maps[1:]:
        if len(curr_map) != len(input_maps[0]):
            parser.error('Not the same number of echoes per contrast')
    if len(input_maps[0]) == 1:
        single_echo = True
    else:
        single_echo = False

    if args.in_B1_map and not args.in_mtoff_t1:
        logging.warning('No B1 correction was applied because no MTsat or '
                        'ihMTsat can be computed without the in_mtoff_t1.')

    if args.B1_correction_method == 'model_based' and not args.B1_fitvalues:
        parser.error('Fitvalues files must be given when choosing the '
                     'model-based B1 correction method. Please use '
                     '--B1_fitvalues.')

    # Set TR and FlipAngle parameters
    if args.in_acq_parameters:
        flip_angles = np.asarray(args.in_acq_parameters[:2]) * np.pi / 180.
        rep_times = np.asarray(args.in_acq_parameters[2:]) * 1000
        if rep_times[0] > 10000 or rep_times[1] > 10000:
            logging.warning('Given repetition times do not seem to be given '
                            'in seconds. MTsat results might be affected.')
    elif args.in_jsons:
        rep_times = []
        flip_angles = []
        for curr_json in args.in_jsons:
            acq_parameter = get_acq_parameters(curr_json,
                                               ['RepetitionTime', 'FlipAngle'])
            if acq_parameter[0] > 10:
                logging.warning('Repetition time found in {} does not seem to '
                                'be given in seconds. MTsat results might be '
                                'affected.'.format(curr_json))
            rep_times.append(acq_parameter[0] * 1000)  # convert ms.
            flip_angles.append(acq_parameter[1] * np.pi / 180.)  # convert rad.

    # Fix issue from the presence of invalide value and division by zero
    np.seterr(divide='ignore', invalid='ignore')

    # Define reference image for saving maps
    affine = nib.load(input_maps[0][0]).affine

    # Load B1 image
    if args.in_B1_map and args.in_mtoff_t1:
        B1_img = nib.load(args.in_B1_map)
        B1_map = B1_img.get_fdata(dtype=np.float32)
        B1_map = adjust_B1_map_intensities(B1_map, nominal=args.B1_nominal)
        B1_map = smooth_B1_map(B1_map, wdims=args.B1_smooth_dims)
        if args.B1_correction_method == 'model_based':
            # Apply the B1 map to the flip angles for model-based correction
            flip_angles[0] *= B1_map
            flip_angles[1] *= B1_map
        if args.extended:
            nib.save(nib.Nifti1Image(B1_map, affine),
                     os.path.join(extended_dir, "B1_map.nii.gz"))

    # Define contrasts maps names
    contrast_names_og = contrast_names
    if args.filtering:
        contrast_names = [curr_name + '_filter'
                          for curr_name in contrast_names]
    if single_echo:
        contrast_names = [curr_name + '_single_echo'
                          for curr_name in contrast_names]
    if args.out_prefix:
        contrast_names = [args.out_prefix + '_' + curr_name
                          for curr_name in contrast_names]

    # Compute contrasts maps
    contrast_maps = []
    for idx, curr_map in enumerate(input_maps):
        input_images = []
        for image in curr_map:
            img, _ = load_img(image)
            input_images.append(img)
        merged_curr_map = concatenate(input_images, input_images[0])
        contrast_maps.append(process_contrast_map(merged_curr_map,
                                                  filtering=args.filtering,
                                                  single_echo=single_echo))
        if args.extended:
            nib.save(nib.Nifti1Image(contrast_maps[idx].astype(np.float32),
                                     affine),
                     os.path.join(extended_dir,
                                  contrast_names[idx] + '.nii.gz'))

    # Compute MTR
    if 'positive' in contrast_names_og and 'negative' in contrast_names_og:
        MTR = compute_ratio_map((contrast_maps[0] + contrast_maps[1]) / 2,
                                contrast_maps[2])
    else:
        MTR = compute_ratio_map(contrast_maps[0], contrast_maps[1])

    img_name = ['MTR']
    img_data = [MTR]

    # Compute MTsat
    if args.in_mtoff_t1:
        MTsat_maps = []
        if 'positive' in contrast_names_og:
            MTsat_sp, T1app = compute_saturation_map(contrast_maps[0],
                                                     contrast_maps[-2],
                                                     contrast_maps[-1],
                                                     flip_angles, rep_times)
            MTsat_maps.append(MTsat_sp)
        if 'negative' in contrast_names_og:
            MTsat_sn, T1app = compute_saturation_map(contrast_maps[-3],
                                                     contrast_maps[-2],
                                                     contrast_maps[-1],
                                                     flip_angles, rep_times)
            MTsat_maps.append(MTsat_sn)
        R1app = 1000 / T1app  # convert 1/ms to 1/s
        if args.extended:
            if 'positive' in contrast_names_og:
                nib.save(nib.Nifti1Image(MTsat_sp, affine),
                         os.path.join(extended_dir, "MTsat_positive.nii.gz"))
            if 'negative' in contrast_names_og:
                nib.save(nib.Nifti1Image(MTsat_sn, affine),
                         os.path.join(extended_dir, "MTsat_negative.nii.gz"))
            nib.save(nib.Nifti1Image(R1app, affine),
                     os.path.join(extended_dir, "apparent_R1.nii.gz"))

        # Apply model-based B1 correction
        if args.in_B1_map and args.B1_correction_method == 'model_based':
            for i, MTsat_map in enumerate(MTsat_maps):
                MTsat_maps[i] = apply_B1_corr_model_based(MTsat_map, B1_map,
                                                          R1app,
                                                          args.B1_fitvalues[i])

        # Compute MTsat and ihMTsat from saturations
        if 'positive' in contrast_names_og and 'negative' in contrast_names_og:
            MTsat = (MTsat_maps[0] + MTsat_maps[1]) / 2
        else:
            MTsat = MTsat_maps[0]

        # Apply empiric B1 correction
        if args.in_B1_map and args.B1_correction_method == 'empiric':
            # MTR = apply_B1_correction_empiric(MTR, B1_map)
            MTsat = apply_B1_corr_empiric(MTsat, B1_map)

        img_name.append('MTsat')
        img_data.append(MTsat)

    # Apply thresholds on maps
    for i, map in enumerate(img_data):
        img_data[i] = threshold_map(map, args.mask, 0, 100)

    # Save ihMT and MT images
    if args.filtering:
        img_name = [curr_name + '_filter'
                    for curr_name in img_name]
    if single_echo:
        img_name = [curr_name + '_single_echo'
                    for curr_name in img_name]
    if args.in_B1_map:
        img_name = [curr_name + '_B1_corrected'
                    for curr_name in img_name]
    if args.out_prefix:
        img_name = [args.out_prefix + '_' + curr_name
                    for curr_name in img_name]

    for img_to_save, name in zip(img_data, img_name):
        nib.save(nib.Nifti1Image(img_to_save.astype(np.float32),
                                 affine),
                 os.path.join(output_dir, name + '.nii.gz'))


if __name__ == '__main__':
    main()
