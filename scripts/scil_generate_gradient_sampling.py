#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate multi-shell gradient sampling with various processing options. Helps
accelerate acquisition, optimize duty cycle and avoid artefacts.

Multi-shell gradient sampling is generated as in [1]. The bvecs are then
flipped to maximize spread for eddy current correction, b0s are interleaved at
equal spacing and the non-b0 samples are finally shuffled to minimize the total
diffusion gradient amplitude over a few TR.
"""

import argparse
import logging
import numpy as np
import os

from scilpy.io.utils import (
    add_overwrite_arg, add_verbose_arg, assert_outputs_exist)
from scilpy.gradients.gen_gradient_sampling import (
    generate_gradient_sampling)
from scilpy.gradientsampling.optimize_gradient_sampling import (
    add_b0s, add_bvalue_b0, compute_bvalue_lin_b, compute_bvalue_lin_q,
    compute_min_duty_cycle_bruteforce, correct_b0s_philips, swap_sampling_eddy)
from scilpy.gradientsampling.save_gradient_sampling import (
    save_gradient_sampling_fsl, save_gradient_sampling_mrtrix)


EPILOG = """
References: [1] Emmanuel Caruyer, Christophe Lenglet, Guillermo Sapiro,
Rachid Deriche. Design of multishell gradient sampling with uniform coverage
in diffusion MRI. Magnetic Resonance in Medicine, Wiley, 2013, 69 (6),
pp. 1534-1540. <http://dx.doi.org/10.1002/mrm.24736>
    """


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog=EPILOG)
    p.add_argument('nb_samples', type=int, nargs='+',
                   help='Number of samples on each non b0 shell. '
                        'If multishell, provide a number per shell.')
    p.add_argument('out_basename',
                   help='Gradient sampling output basename (don\'t '
                        'include extension).\n'
                        'Please add options --fsl and/or --mrtrix below.')

    p.add_argument('--eddy', action='store_true',
                   help='If set, we apply eddy optimization.\nB-vectors are '
                        'flipped to be well spread without symmetry.')
    p.add_argument('--duty', action='store_true',
                   help='If set, we apply duty cycle optimization. '
                        '\nB-vectors are shuffled to reduce consecutive '
                        'colinearity in the samples. [%(default)s]')
    p.add_argument('--b0_every',
                   type=int, default=-1,
                   help='Interleave a b0 every b0_every. \nNo b0 if 0. '
                        '\nOnly one b0 at the beginning if > number of '
                        'samples or if -1 (default).')
    p.add_argument('--b0_end', action='store_true',
                   help='If set, adds a b0 as last sample.')
    p.add_argument('--b0_value', type=float, default=0.0,
                   help='b-value of the b0s. [%(default)s]')
    p.add_argument('--b0_philips', action='store_true',
                   help='If set, replaces values of b0s bvecs by existing '
                        'bvecs for Philips handling.')

    bvals_group = p.add_mutually_exclusive_group(required=True)
    bvals_group.add_argument('--bvals', type=float, nargs='+', metavar='bvals',
                             help='bval of each non-b0 shell.')
    bvals_group.add_argument('--b_lin_max', type=float,
                             help='b-max for linear bval distribution '
                                  'in *b*. [replaces -bvals]')
    bvals_group.add_argument('--q_lin_max', type=float,
                             help='b-max for linear bval distribution '
                                  'in *q*. [replaces -bvals]')

    g1 = p.add_argument_group(title='Save as')
    g1 = g1.add_mutually_exclusive_group(required=True)
    g1.add_argument('--fsl', action='store_true',
                    help='Save in FSL format (.bvec/.bval).')
    g1.add_argument('--mrtrix', action='store_true',
                    help='Save in MRtrix format (.b).')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    out_basename, _ = os.path.splitext(args.out_basename)

    if args.fsl:
        out_filename = [out_basename + '.bval', out_basename + '.bvec']
    else:  # args.mrtrix = True
        out_filename = out_basename + '.b'

    assert_outputs_exist(parser, args, out_filename)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(log_level)

    # Total number of samples
    K = np.sum(args.nb_samples)
    # Number of non-b0 shells
    S = len(args.nb_samples)

    b0_every = args.b0_every
    b0_end = args.b0_end
    b0_value = args.b0_value
    b0_philips = args.b0_philips

    # Only a b0 at the beginning
    if (b0_every > K) or (b0_every < 0):
        b0_every = K + 1

    # Compute bval list
    if args.bvals is not None:
        bvals = args.bvals
        unique_bvals = np.unique(bvals)
        if len(unique_bvals) != S:
            parser.error('You have provided {} shells '.format(S) +
                         'but {} unique bvals.'.format(len(unique_bvals)))

    elif args.b_lin_max is not None:
        bvals = compute_bvalue_lin_b(bmin=0.0, bmax=args.b_lin_max,
                                     nb_of_b_inside=S - 1, exclude_bmin=True)
    elif args.q_lin_max is not None:
        bvals = compute_bvalue_lin_q(bmin=0.0, bmax=args.q_lin_max,
                                     nb_of_b_inside=S - 1, exclude_bmin=True)
    # Add b0 b-value
    if b0_every != 0:
        bvals = add_bvalue_b0(bvals, b0_value=b0_value)

    # Gradient sampling generation
    points, shell_idx = generate_gradient_sampling(args.nb_samples, verbose=int(
        3 - logging.getLogger().getEffectiveLevel()//10))

    # eddy current optimization
    if args.eddy:
        points, shell_idx = swap_sampling_eddy(points, shell_idx)

    # Adding interleaved b0s
    if b0_every != 0:
        points, shell_idx = add_b0s(points,
                                    shell_idx,
                                    b0_every=b0_every,
                                    finish_b0=b0_end)

    # duty cycle optimization
    if args.duty:
        points, shell_idx = compute_min_duty_cycle_bruteforce(
            points, shell_idx, bvals)

    # Correcting b0s bvecs for Philips
    if b0_philips and np.sum(shell_idx == -1) > 1:
        points, shell_idx = correct_b0s_philips(points, shell_idx)

    if args.fsl:
        save_gradient_sampling_fsl(points, shell_idx, bvals,
                                   out_filename[0], out_filename[1])
    else:  # args.mrtrix:
        if not points.shape[0] == 3:
            points = points.transpose()
            save_gradient_sampling_mrtrix(points, shell_idx, bvals,
                                          filename=out_filename)


if __name__ == "__main__":
    main()
