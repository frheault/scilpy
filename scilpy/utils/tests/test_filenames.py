import pytest
from scilpy.utils.filenames import add_filename_suffix, split_name_with_nii


def test_add_filename_suffix():
    """Test the add_filename_suffix function."""
    # Test with a standard .nii.gz file
    filename = "test.nii.gz"
    suffix = "_new"
    expected = "test_new.nii.gz"
    assert add_filename_suffix(filename, suffix) == expected

    # Test with a file that has no extension
    filename = "test"
    suffix = "_new"
    expected = "test_new"
    assert add_filename_suffix(filename, suffix) == expected

    # Test with a single extension
    filename = "test.nii"
    suffix = "_new"
    expected = "test_new.nii"
    assert add_filename_suffix(filename, suffix) == expected

    # Test with multiple dots in the filename
    filename = "test.v1.2.nii.gz"
    suffix = "_new"
    expected = "test.v1.2_new.nii.gz"
    assert add_filename_suffix(filename, suffix) == expected

    # Test with an empty suffix
    filename = "test.nii.gz"
    suffix = ""
    expected = "test.nii.gz"
    assert add_filename_suffix(filename, suffix) == expected


def test_split_name_with_nii():
    """Test the split_name_with_nii function."""
    # Test with .nii.gz
    filename = "test.nii.gz"
    base, ext = split_name_with_nii(filename)
    assert base == "test"
    assert ext == ".nii.gz"

    # Test with .nii
    filename = "test.nii"
    base, ext = split_name_with_nii(filename)
    assert base == "test"
    assert ext == ".nii"

    # Test with .gz
    filename = "test.gz"
    base, ext = split_name_with_nii(filename)
    assert base == "test"
    assert ext == ".gz"

    # Test with no extension
    filename = "test"
    base, ext = split_name_with_nii(filename)
    assert base == "test"
    assert ext == ""

    # Test with a hidden file
    filename = ".test.nii.gz"
    base, ext = split_name_with_nii(filename)
    assert base == ".test"
    assert ext == ".nii.gz"

    # Test with multiple dots
    filename = "a.b.c.nii.gz"
    base, ext = split_name_with_nii(filename)
    assert base == "a.b.c"
    assert ext == ".nii.gz"
