import numpy as np
from scilpy.io.stateful_image import StatefulImage
from scilpy.io.stateful_gradient import StatefulGradient


def test_stateful_gradient_ras():
    affine = np.eye(4)
    data = np.zeros((10, 10, 10))
    simg = StatefulImage(data, affine)
    simg._original_affine = affine.copy()

    bvals = np.array([0, 1000])
    bvecs = np.array([[0, 0, 0], [1, 0, 0]])
    sgrad = StatefulGradient(bvals, bvecs, simg)
    assert np.allclose(sgrad.bvecs[1], [1, 0, 0])


def test_stateful_gradient_las():
    affine_las = np.diag([-1, 1, 1, 1])
    data = np.zeros((10, 10, 10))
    simg_las = StatefulImage(data, affine_las)
    simg_las._original_affine = affine_las.copy()

    bvals = np.array([0, 1000])
    bvecs = np.array([[0, 0, 0], [1, 0, 0]])
    sgrad_las = StatefulGradient(bvals, bvecs, simg_las)
    # FSL [1,0,0] for LAS should be [1,0,0] in World because of determinant flip
    assert np.allclose(sgrad_las.bvecs[1], [1, 0, 0])


def test_stateful_gradient_reorientation():
    affine_ras = np.eye(4)
    data = np.zeros((10, 10, 10))
    simg = StatefulImage(data, affine_ras)
    simg._original_affine = affine_ras.copy()

    bvals = np.array([0, 1000])
    bvecs = np.array([[0, 0, 0], [1, 0, 0]])
    sgrad = StatefulGradient(bvals, bvecs, simg)

    # Reorient simg to LAS
    simg.reorient("LAS")

    # Gradients should still point Right in RAS mm
    assert np.allclose(sgrad.bvecs[1], [1, 0, 0])

    # But if we ask for bvecs in the NEW axes (LAS)
    bvecs_las = sgrad.get_bvecs_reoriented(simg.affine)
    # In FSL/MRtrix convention, for LAS, [1, 0, 0] in the file means Right.
    assert np.allclose(bvecs_las[1], [1, 0, 0])


def test_stateful_gradient_normalization():
    affine = np.eye(4)
    data = np.zeros((10, 10, 10))
    simg = StatefulImage(data, affine)
    simg._original_affine = affine.copy()

    bvals = np.array([0, 1000])
    bvecs = np.array([[0, 0, 0], [2, 0, 0]])  # Non-normalized
    sgrad = StatefulGradient(bvals, bvecs, simg, normalize=True)
    assert np.allclose(sgrad.bvecs[1], [1, 0, 0])


def test_stateful_gradient_oblique():
    # Affine with 45 deg rotation around Z
    cos45 = np.sqrt(2) / 2
    affine_rot = np.array([
        [cos45, -cos45, 0, 0],
        [cos45, cos45, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    data = np.zeros((10, 10, 10))
    simg = StatefulImage(data, affine_rot)
    simg._original_affine = affine_rot.copy()

    bvals = np.array([0, 1000])
    bvecs = np.array([[0, 0, 0], [1, 0, 0]])
    sgrad = StatefulGradient(bvals, bvecs, simg)

    # Rotation should be ignored. Dominant axes are RAS.
    # [1, 0, 0] in FSL (along image axis 0) should be [1, 0, 0] in Canonical RAS.
    assert np.allclose(sgrad.bvecs[1], [1, 0, 0])

    # Even if we ask to reorient to a specific affine (the same one)
    # it should still be [1, 0, 0] because it's voxel-aligned.
    bvecs_rot = sgrad.get_bvecs_reoriented(affine_rot)
    assert np.allclose(bvecs_rot[1], [1, 0, 0])
