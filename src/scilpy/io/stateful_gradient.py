
import logging
import os
import numpy as np
from nibabel.orientations import aff2axcodes
from dipy.io import read_bvals_bvecs

from scilpy.io.stateful_image import StatefulImage

class StatefulGradient:
    """
    Class to handle diffusion gradients (bvals/bvecs) in a stateful manner,
    synchronized with a StatefulImage.

    Internally, bvecs are stored in RAS mm (World/Scanner space).
    This ensures that gradients are orientation-invariant in memory.
    The transformation from FSL (axis-relative) to RAS mm (world-relative)
    follows the MRtrix convention, accounting for the image affine.
    """

    def __init__(self, bvals, bvecs, simg, space='fsl', normalize=True):
        """
        Parameters
        ----------
        bvals: np.ndarray
            1D array of b-values.
        bvecs: np.ndarray
            Nx3 array of gradient directions.
        simg: StatefulImage
            The reference image these gradients are associated with.
        space: str
            The coordinate space of the input bvecs.
            'fsl': Gradients are defined relative to the image axes.
                   This transformation uses the CURRENT affine of the simg.
            'rasmm': Gradients are already in RAS mm (World space).
        normalize: bool
            If True, bvecs will be normalized to unit length.
        """
        if not isinstance(simg, StatefulImage):
            raise TypeError("Reference image must be a StatefulImage instance.")

        self._bvals = np.asarray(bvals)
        self._simg = simg

        bvecs = np.asarray(bvecs, dtype=np.float64)
        if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
            bvecs = bvecs.T

        if normalize:
            norms = np.linalg.norm(bvecs, axis=1)
            # Avoid division by zero for b0s
            idx = norms > 0
            bvecs[idx] /= norms[idx, None]

        if space.lower() == 'fsl':
            self._bvecs = self._axes_to_rasmm(bvecs, simg.affine)
        elif space.lower() == 'rasmm':
            self._bvecs = bvecs
        else:
            raise ValueError("Space must be 'fsl' or 'rasmm'.")

    @property
    def bvals(self):
        return self._bvals

    @property
    def bvecs(self):
        """Returns bvecs in the internal RAS mm space."""
        return self._bvecs

    @property
    def simg(self):
        return self._simg

    def to_rasmm(self):
        """Alias for clarity, returning internal World-space bvecs."""
        return self._bvecs

    def get_bvecs_reoriented(self, reference_image_or_affine):
        """
        Projects the internal RAS mm bvecs back into a specific axis space.

        Parameters
        ----------
        reference_image_or_affine: StatefulImage | nib.Nifti1Image | np.ndarray
            The target orientation defined by a StatefulImage or a 4x4 affine.

        Returns
        -------
        np.ndarray: Bvecs oriented relative to the target's axes (FSL format).
        """
        if hasattr(reference_image_or_affine, 'affine'):
            affine = reference_image_or_affine.affine
        elif isinstance(reference_image_or_affine, np.ndarray) and \
                reference_image_or_affine.shape == (4, 4):
            affine = reference_image_or_affine
        else:
            raise TypeError("Reference must be a StatefulImage, Nifti1Image, "
                            "or a 4x4 affine.")

        return self._rasmm_to_axes(self._bvecs, affine)

    def _get_fsl_rotation(self, affine):
        """
        Computes the rotation matrix R used by FSL to relate axis-space
        to world-space.
        """
        R = affine[:3, :3].copy()
        norms = np.linalg.norm(R, axis=0)
        R /= norms

        # FSL's implicit flip for left-handed coordinate systems:
        # If the determinant is negative, the first axis (x) is flipped
        # to maintain a right-handed system in the bvecs.
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        
        return R

    def _axes_to_rasmm(self, bvecs, affine):
        """Transforms bvecs from Axis space (FSL) to RAS mm (World)."""
        R = self._get_fsl_rotation(affine)
        # v_world = R @ v_fsl
        return (R @ bvecs.T).T

    def _rasmm_to_axes(self, bvecs, affine):
        """Transforms bvecs from RAS mm (World) to Axis space (FSL)."""
        R = self._get_fsl_rotation(affine)
        # v_fsl = inv(R) @ v_world. Since R is orthogonal, inv(R) = R.T
        return (R.T @ bvecs.T).T

    @classmethod
    def load(cls, bval_file, bvec_file, simg, normalize=True):
        """
        Loads bvals/bvecs from disk and associates them with a StatefulImage.
        Automatically detects format (currently FSL only supported).
        """
        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
        if bvals is None:
            # Create dummy bvals if not provided
            bvals = np.zeros(len(bvecs))
        return cls(bvals, bvecs, simg, space='fsl', normalize=normalize)

    def save(self, bval_path, bvec_path):
        """
        Saves bvals and bvecs to disk in FSL format.
        Automatically uses the reference image's ORIGINAL affine to ensure
        saved files are synchronized with the saved (stride-preserved) image.
        """
        # FSL bvals are saved as a single row
        np.savetxt(bval_path, self._bvals[None, :], fmt='%d')
        # Project to original axes space
        bvecs_fsl = self.get_bvecs_reoriented(self._simg.original_affine)
        # FSL bvecs are saved as 3 rows (3, N)
        np.savetxt(bvec_path, bvecs_fsl.T, fmt='%.8f')

    def __repr__(self):
        return (f"<StatefulGradient: {len(self._bvals)} gradients, "
                f"Ref: {aff2axcodes(self._simg.affine)}>")
