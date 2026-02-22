# -*- coding: utf-8 -*-

import logging
import os

import numpy as np
from dipy.io import read_bvals_bvecs as dipy_read_bvals_bvecs

from scilpy.io.stateful_gradient import StatefulGradient


def read_bvals_bvecs(bval_file, bvec_file, simg=None):
    """
    Polymorphic loader for gradient tables.

    If simg is None, it behaves like dipy.io.read_bvals_bvecs and returns
    raw numpy arrays.
    If simg is a StatefulImage, it returns a StatefulGradient object
    which is synchronized with the image orientation.

    Parameters
    ----------
    bval_file: str
        Path to the bvals file.
    bvec_file: str
        Path to the bvecs file.
    simg: StatefulImage, optional
        Reference stateful image.

    Returns
    -------
    (bvals, bvecs) | StatefulGradient
    """
    if simg is None:
        return dipy_read_bvals_bvecs(bval_file, bvec_file)

    return StatefulGradient.load(bval_file, bvec_file, simg)


def fsl2mrtrix(fsl_bval_filename, fsl_bvec_filename, mrtrix_filename, simg=None):
    """
    Convert a fsl dir_grad.bvec/.bval files to mrtrix encoding.b file.
    Saves the result.

    Parameters
    ----------
    fsl_bval_filename: str
        path to input fsl bval file.
    fsl_bvec_filename: str
        path to input fsl bvec file.
    mrtrix_filename : str
        path to output mrtrix encoding.b file.
    simg: StatefulImage, optional
        Reference image to handle affine-aware bvec conversion.
        If None, a standard RAS identity is assumed.
    """
    if simg:
        sgrad = StatefulGradient.load(fsl_bval_filename, fsl_bvec_filename, simg)
    else:
        # Create a dummy StatefulImage with identity affine
        from scilpy.io.stateful_image import StatefulImage
        dummy_simg = StatefulImage(np.zeros((1, 1, 1)), np.eye(4))
        dummy_simg._original_affine = np.eye(4)
        sgrad = StatefulGradient.load(fsl_bval_filename, fsl_bvec_filename,
                                      dummy_simg)

    bvals = sgrad.bvals
    bvecs = sgrad.to_ras()  # MRtrix uses World/RAS mm coordinates

    # bvecs are Nx3, need to be 3xN for save_gradient_sampling_mrtrix?
    # Let's check save_gradient_sampling_mrtrix

    unique_bvals = np.unique(bvals).tolist()
    shell_idx = [int(np.where(bval == unique_bvals)[0][0]) for bval in bvals]

    # Remove .bval and .bvec if present
    mrtrix_filename = mrtrix_filename.replace('.b', '')

    save_gradient_sampling_mrtrix(bvecs.T, shell_idx, unique_bvals,
                                  mrtrix_filename + '.b')


def mrtrix2fsl(mrtrix_filename, fsl_filename, simg=None):
    """
    Convert a mrtrix encoding.b file to fsl dir_grad.bvec/.bval files.
    Saves the result.

    Parameters
    ----------
    mrtrix_filename : str
        path to mrtrix encoding.b file.
    fsl_filename: str
        path to the output fsl files. Files will be named
        fsl_bval_filename.bval and fsl_bval_filename.bvec.
    simg: StatefulImage, optional
        Reference image to handle affine-aware bvec conversion.
        If None, a standard RAS identity is assumed.
    """
    # Remove .bval and .bvec if present
    fsl_filename = fsl_filename.replace('.bval', '')
    fsl_filename = fsl_filename.replace('.bvec', '')

    mrtrix_b = np.loadtxt(mrtrix_filename)
    if not len(mrtrix_b.shape) == 2 or not mrtrix_b.shape[1] == 4:
        raise ValueError('mrtrix file must have 4 columns')

    points_world = mrtrix_b[:, 0:3]
    shells = mrtrix_b[:, 3]

    if simg is None:
        from scilpy.io.stateful_image import StatefulImage
        simg = StatefulImage(np.zeros((1, 1, 1)), np.eye(4))
        simg._original_affine = np.eye(4)

    sgrad = StatefulGradient(shells, points_world, simg, space='ras')

    # Save uses the original affine by default
    sgrad.save(fsl_filename + '.bval', fsl_filename + '.bvec')


def save_gradient_sampling_mrtrix(bvecs, shell_idx, bvals, filename):
    """
    Save table gradient (MRtrix format)

    Parameters
    ----------
    bvecs: numpy.array
        bvecs normalized to 1.
    shell_idx: numpy.array
        Shell index for bvecs.
    bvals: numpy.array
    filename: str
        output file name
    ------
    """
    with open(filename, 'w') as f:
        for idx in range(bvecs.shape[1]):
            f.write('{:.8f} {:.8f} {:.8f} {:}\n'
                    .format(bvecs[0, idx], bvecs[1, idx], bvecs[2, idx],
                            bvals[shell_idx[idx]]))

    logging.info('Gradient sampling saved in MRtrix format as {}'
                 .format(filename))


def save_gradient_sampling_fsl(bvecs, shell_idx, bvals, filename_bval,
                               filename_bvec):
    """
    Save table gradient (FSL format)

    Parameters
    ----------
    bvecs: numpy.array
        bvecs normalized to 1.
    shell_idx: numpy.array
        Shell index for bvecs.
    bvals: numpy.array
    filename_bval: str
        output bval filename.
    filename_bvec: str
        output bvec filename.
    ------
    """
    basename, ext = os.path.splitext(filename_bval)

    np.savetxt(filename_bvec, bvecs, fmt='%.8f')
    np.savetxt(filename_bval,
               np.array([bvals[idx] for idx in shell_idx])[None, :],
               fmt='%.3f')

    logging.info('Gradient sampling saved in FSL format as {}'
                 .format(basename + '{.bvec/.bval}'))
