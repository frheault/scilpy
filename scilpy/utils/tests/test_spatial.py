import numpy as np
import nibabel as nib
from numpy.testing import assert_array_equal, assert_almost_equal

from scilpy.utils.spatial import (
    _any2ras_index,
    get_axis_name,
    get_coordinate_name,
    get_basis_vector_name,
    get_axis_index,
    voxel_to_world,
    compute_distance_barycenters,
    WorldBoundingBox,
    world_to_voxel,
    generate_rotation_matrix,
)


def test_any2ras_index():
    """Test the _any2ras_index function."""
    # Test with identity affine
    affine = np.eye(4)
    ix, sgn = _any2ras_index(0, affine)
    assert ix == 0 and sgn == ""
    ix, sgn = _any2ras_index(1, affine)
    assert ix == 1 and sgn == ""
    ix, sgn = _any2ras_index(2, affine)
    assert ix == 2 and sgn == ""

    # Test with flipped affine
    affine = np.array([[-1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])
    ix, sgn = _any2ras_index(0, affine)
    assert ix == 0 and sgn == "-"
    ix, sgn = _any2ras_index(1, affine)
    assert ix == 1 and sgn == "-"
    ix, sgn = _any2ras_index(2, affine)
    assert ix == 2 and sgn == "-"


def test_get_axis_name():
    """Test the get_axis_name function."""
    affine = np.eye(4)
    assert get_axis_name(0, affine) == "sagittal"
    assert get_axis_name(1, affine) == "coronal"
    assert get_axis_name(2, affine) == "axial"


def test_get_coordinate_name():
    """Test the get_coordinate_name function."""
    affine = np.eye(4)
    assert get_coordinate_name(0, affine) == "x"
    assert get_coordinate_name(1, affine) == "y"
    assert get_coordinate_name(2, affine) == "z"


def test_get_basis_vector_name():
    """Test the get_basis_vector_name function."""
    affine = np.eye(4)
    assert get_basis_vector_name(0, affine) == "i"
    assert get_basis_vector_name(1, affine) == "j"
    assert get_basis_vector_name(2, affine) == "k"


def test_get_axis_index():
    """Test the get_axis_index function."""
    affine = np.eye(4)
    assert get_axis_index("sagittal", affine) == 0
    assert get_axis_index("coronal", affine) == 1
    assert get_axis_index("axial", affine) == 2
    assert get_axis_index("x", affine) == 0
    assert get_axis_index("y", affine) == 1
    assert get_axis_index("z", affine) == 2
    assert get_axis_index("i", affine) == 0
    assert get_axis_index("j", affine) == 1
    assert get_axis_index("k", affine) == 2


def test_voxel_to_world():
    """Test the voxel_to_world function."""
    affine = np.array([[2, 0, 0, 10],
                       [0, 2, 0, 20],
                       [0, 0, 2, 30],
                       [0, 0, 0, 1]])
    coord = np.array([1, 2, 3])
    world_coord = voxel_to_world(coord, affine)
    assert_array_equal(world_coord, [12, 24, 36])


def test_compute_distance_barycenters():
    """Test the compute_distance_barycenters function."""
    ref1 = nib.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4))
    ref2 = nib.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4))
    transfo = np.array([[1, 0, 0, 10],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    dist_before, dist_after = compute_distance_barycenters(ref1, ref2, transfo)
    assert_almost_equal(dist_before, 0.0)
    assert_almost_equal(dist_after, 10.0)


def test_world_bounding_box_constructor():
    """Test the WorldBoundingBox constructor."""
    minimums = np.array([-10, -10, -10])
    maximums = np.array([10, 10, 10])
    voxel_size = np.array([1, 1, 1])
    bbox = WorldBoundingBox(minimums, maximums, voxel_size)
    assert_array_equal(bbox.minimums, minimums)
    assert_array_equal(bbox.maximums, maximums)
    assert_array_equal(bbox.voxel_size, voxel_size)


def test_world_to_voxel():
    """Test the world_to_voxel function."""
    affine = np.array([[2, 0, 0, 10],
                       [0, 2, 0, 20],
                       [0, 0, 2, 30],
                       [0, 0, 0, 1]])
    coord = np.array([12, 24, 36])
    vox_coord = world_to_voxel(coord, affine)
    assert_array_equal(vox_coord, [1, 2, 3])


def test_generate_rotation_matrix():
    """Test the generate_rotation_matrix function."""
    angles = [np.pi/2, 0, 0]
    trans = [10, 20, 30]
    rot = generate_rotation_matrix(angles, trans)
    expected_rot = np.array([[1, 0, 0, 10],
                             [0, 0, -1, 20],
                             [0, 1, 0, 30],
                             [0, 0, 0, 1]])
    assert_almost_equal(rot, expected_rot)
