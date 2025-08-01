import pytest
from scilpy.io.deprecator import deprecate_script, ScilpyExpiredDeprecation


def test_deprecate_script():
    # Mock the version to be the same as the from_version
    import scilpy
    scilpy.version._version_major = 1
    scilpy.version._version_minor = 0

    # Test that a warning is issued
    @deprecate_script("test_script", "This is a test.", "1.0")
    def dummy_function():
        return "Hello"

    with pytest.warns(DeprecationWarning):
        assert dummy_function() == "Hello"

    # Test that the decorator raises an error for an expired version
    @deprecate_script("test_script", "This is a test.", "0.1")
    def expired_function():
        return "Hello"

    # Mock the version to be higher than the expiration
    scilpy.version._version_major = 1
    scilpy.version._version_minor = 0

    with pytest.raises(ScilpyExpiredDeprecation):
        expired_function()

    # Reset the version
    scilpy.version._version_major = 2
    scilpy.version._version_minor = 0
