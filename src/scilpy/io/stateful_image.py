# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
from dipy.io.utils import get_reference_info
from scilpy.utils.orientation import validate_voxel_order


class StatefulImage(nib.Nifti1Image):
    """
    A class that extends nib.Nifti1Image to manage image orientation state.

    This class ensures that image data loaded into memory is always in a
    consistent orientation (RAS by default), while preserving the original
    on-disk orientation information. When saving, the image is automatically
    reverted to its original orientation, ensuring non-destructive operations.
    """

    def __init__(self, dataobj, affine, header=None, extra=None,
                 file_map=None, original_affine=None,
                 original_dimensions=None, original_voxel_sizes=None,
                 original_axcodes=None):
        """
        Initialize a StatefulImage object.

        Extends the Nifti1Image constructor to store original orientation info.
        """
        super().__init__(dataobj, affine, header, extra, file_map)

        # Store original image information
        self._original_affine = original_affine
        self._original_dimensions = original_dimensions
        self._original_voxel_sizes = original_voxel_sizes
        self._original_axcodes = original_axcodes

    @property
    def original_affine(self):
        return self._original_affine

    @property
    def original_axcodes(self):
        return self._original_axcodes

    @property
    def axcodes(self):
        return nib.orientations.aff2axcodes(self.affine)

    @classmethod
    def load(cls, filename, to_orientation="RAS"):
        """
        Load a NIfTI image, store its original orientation, and reorient it.

        Parameters
        ----------
        filename : str
            Path to the NIfTI file.
        to_orientation : str or tuple, optional
            The target orientation for the in-memory data. Default is "RAS".

        Returns
        -------
        StatefulImage
            An instance of StatefulImage with data in the target orientation.
        """
        img = nib.load(filename)

        original_affine = img.affine.copy()
        original_axcodes = nib.orientations.aff2axcodes(img.affine)
        original_dims = img.header.get_data_shape()
        original_voxel_sizes = img.header.get_zooms()

        simg = cls(img.dataobj, img.affine, img.header,
                   original_affine=original_affine,
                   original_dimensions=original_dims,
                   original_voxel_sizes=original_voxel_sizes,
                   original_axcodes=original_axcodes)

        if to_orientation:
            simg.reorient(to_orientation)

        return simg

    @classmethod
    def create_from(cls, nib_img, reference_simg):
        """
        Create a StatefulImage from a standard nibabel image, using metadata
        from a reference StatefulImage.

        This is the preferred way to create new images from derived data
        while maintaining the original stride information for saving.

        Parameters
        ----------
        nib_img : nibabel.spatialimages.SpatialImage
            The new image data.
        reference_simg : StatefulImage
            The image to copy original orientation metadata from.

        Returns
        -------
        StatefulImage
            A new StatefulImage instance.
        """
        return cls(nib_img.dataobj, nib_img.affine, nib_img.header,
                   original_affine=reference_simg._original_affine,
                   original_dimensions=reference_simg._original_dimensions,
                   original_voxel_sizes=reference_simg._original_voxel_sizes,
                   original_axcodes=reference_simg._original_axcodes)

    def save(self, filename):
        """
        Save the image to disk, reverting to the original orientation.

        Parameters
        ----------
        filename : str
            The path where the image will be saved.
        """
        # Revert to original orientation before saving
        if self._original_axcodes:
            # We must use a copy or a new object to not modify the current state
            data = self.get_fdata()
            img = nib.Nifti1Image(data, self.affine, self.header)
            
            current_axcodes = nib.orientations.aff2axcodes(self.affine)
            start_ornt = nib.orientations.axcodes2ornt(current_axcodes)
            target_ornt = nib.orientations.axcodes2ornt(self._original_axcodes)
            transform = nib.orientations.ornt_transform(start_ornt,
                                                        target_ornt)
            
            img_to_save = img.as_reoriented(transform)
        else:
            img_to_save = self

        nib.save(img_to_save, filename)

    def to_original_orientation(self):
        """
        Reorient the in-memory data to match the original on-disk orientation.

        This method modifies the image in place. It does not return a new
        Nifti1Image instance.

        Raises
        ------
        ValueError
            If the original axis codes are not set.
        """
        if self._original_axcodes is None:
            raise ValueError(
                "Original axis codes are not set cannot reorient to original"
                "orientation.")
        self.reorient(self._original_axcodes)

    def reorient(self, target_axcodes):
        """
        Reorient the in-memory image to a target orientation.

        Parameters
        ----------
        target_axcodes : str or tuple
            The target orientation axis codes (e.g., "LPS", ("R", "A", "S")).
        """
        # Validate only 3 spatial codes
        target_axcodes = validate_voxel_order(target_axcodes)

        current_axcodes = nib.orientations.aff2axcodes(self.affine)
        if current_axcodes == tuple(target_axcodes):
            return

        start_ornt = nib.orientations.axcodes2ornt(current_axcodes)
        target_ornt = nib.orientations.axcodes2ornt(target_axcodes)
        transform = nib.orientations.ornt_transform(start_ornt, target_ornt)

        # Apply reorientation. nibabel handles 4D data by reorienting 
        # the first 3 dimensions when a 3x2 orientation matrix is provided.
        reoriented_img = self.as_reoriented(transform)
        
        # Update self with new data while keeping original orientation info
        self.__init__(reoriented_img.dataobj, reoriented_img.affine,
                      reoriented_img.header,
                      original_affine=self._original_affine,
                      original_dimensions=self._original_dimensions,
                      original_voxel_sizes=self._original_voxel_sizes,
                      original_axcodes=self._original_axcodes)

    def to_ras(self):
        """Convenience method to reorient in-memory data to RAS."""
        self.reorient(("R", "A", "S"))

    def to_lps(self):
        """Convenience method to reorient in-memory data to LPS."""
        self.reorient(("L", "P", "S"))

    def to_reference(self, obj):
        """
        Reorient the in-memory image to match the orientation of a reference
        object.

        Parameters
        ----------
        obj : StatefulImage or nibabel image
            The reference object.
        """
        ref_info = get_reference_info(obj)
        target_axcodes = ref_info[3]
        self.reorient(target_axcodes)
