import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes
from dipy.io import read_bvals_bvecs

from scilpy.io.stateful_image import StatefulImage


class StatefulGradient:
    """
    Class to handle diffusion gradients (bvals/bvecs) in a stateful manner,
    synchronized with a StatefulImage.

    Internally, bvecs are stored in a Canonical RAS space (voxel-aligned).
    This ensures that gradients are orientation-invariant in memory while
    ignoring any non-orthogonal components (like scanner tilt/rotation)
    that might be present in the image affine.
    The transformation from FSL (axis-relative) to Canonical RAS
    follows the FSL/MRtrix conventions for axis flips and swaps.
    """

    def __init__(self, bvals, bvecs, simg, space='fsl', normalize=True,
                 use_original_affine=False):
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
            'ras': Gradients are in RAS space (Canonical RAS).
        normalize: bool
            If True, bvecs will be normalized to unit length.
        use_original_affine: bool
            If True and space='fsl', uses simg.original_affine for conversion.
            Otherwise uses simg.affine.
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
            affine = simg.original_affine if use_original_affine else simg.affine
            self._bvecs = self._axes_to_ras(bvecs, affine)
        elif space.lower() == 'ras':
            self._bvecs = bvecs
        else:
            raise ValueError("Space must be 'fsl' or 'ras'.")

    @property
    def bvals(self):
        return self._bvals

    @property
    def bvecs(self):
        """Returns bvecs in the internal Canonical RAS space (voxel-aligned)."""
        return self._bvecs

    @property
    def simg(self):
        return self._simg

    def to_ras(self):
        """Alias for clarity, returning internal reoriented bvecs."""
        return self._bvecs

    def get_bvecs_reoriented(self, reference_image_or_affine):
        """
        Projects the internal Canonical RAS bvecs back into a specific axis space.

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

        return self._ras_to_axes(self._bvecs, affine)

    def _get_fsl_rotation(self, affine):
        """
        Computes the permutation/flip matrix R used by FSL to relate axis-space
        to a canonical RAS space. Only handles flips and swaps (no rotation).
        """
        ornt = nib.orientations.io_orientation(affine)
        R = np.zeros((3, 3))
        for i, (col, flip) in enumerate(ornt):
            R[int(col), i] = flip

        # FSL's implicit flip for left-handed coordinate systems:
        # If the determinant is negative, the first axis (x) is flipped
        # to maintain a right-handed system in the bvecs.
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1

        return R

    def _axes_to_ras(self, bvecs, affine):
        """Transforms bvecs from Axis space (FSL) to Canonical RAS."""
        R = self._get_fsl_rotation(affine)
        # v_world = R @ v_fsl
        return (R @ bvecs.T).T

    def _ras_to_axes(self, bvecs, affine):
        """Transforms bvecs from Canonical RAS to Axis space (FSL)."""
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

        # When loading from disk, bvecs are assumed relative to the
        # ORIGINAL on-disk orientation of the image.
        return cls(bvals, bvecs, simg, space='fsl', normalize=normalize,
                   use_original_affine=True)

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
