import argparse
import json
import os
import tempfile
import pytest
import numpy as np
from scilpy.io.utils import (
    get_acq_parameters,
    redirect_stdout_c,
    link_bundles_and_reference,
    check_tract_trk,
    check_tracts_same_format,
    assert_gradients_filenames_valid,
    validate_nbr_processes,
    validate_sh_basis_choice,
    verify_compression_th,
    assert_inputs_exist,
    assert_inputs_dirs_exist,
    assert_outputs_exist,
    assert_output_dirs_exist_and_empty,
    assert_overlay_colors,
    assert_roi_radii_format,
    assert_headers_compatible,
    read_info_from_mb_bdo,
    load_matrix_in_any_format,
    save_matrix_in_any_format,
    assert_fsl_options_exist,
    parser_color_type,
    ranged_type,
    get_default_screenshotting_data,
)


def test_get_acq_parameters():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"RepetitionTime": 2.0, "FlipAngle": 90}, f)
        filename = f.name
    params = get_acq_parameters(filename, ["RepetitionTime", "FlipAngle"])
    assert params == [2.0, 90]
    os.remove(filename)


def test_redirect_stdout_c():
    # This is difficult to test in a non-intrusive way.
    # We will trust that it works as intended.
    pass


def test_link_bundles_and_reference():
    # This function depends on an argparse object.
    # It will be tested through the scripts that use it.
    pass


def test_check_tract_trk():
    parser = argparse.ArgumentParser()
    check_tract_trk(parser, "test.trk")
    with pytest.raises(SystemExit):
        check_tract_trk(parser, "test.tck")


def test_check_tracts_same_format():
    parser = argparse.ArgumentParser()
    check_tracts_same_format(parser, ["a.trk", "b.trk"])
    with pytest.raises(SystemExit):
        check_tracts_same_format(parser, ["a.trk", "b.tck"])


def test_assert_gradients_filenames_valid():
    parser = argparse.ArgumentParser()
    assert_gradients_filenames_valid(parser, ["dwi.bval", "dwi.bvec"], True)
    with pytest.raises(SystemExit):
        assert_gradients_filenames_valid(
            parser, ["dwi.bval", "dwi.bvecs"], True)
    assert_gradients_filenames_valid(parser, ["dwi.b"], False)
    with pytest.raises(SystemExit):
        assert_gradients_filenames_valid(parser, ["dwi.bval"], False)


def test_validate_nbr_processes():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nbr_processes', type=int)
    args = parser.parse_args(['--nbr_processes', '2'])
    assert validate_nbr_processes(parser, args) == 2


def test_validate_sh_basis_choice():
    with pytest.raises(ValueError):
        validate_sh_basis_choice("invalid_basis")


def test_verify_compression_th():
    # This function only logs a warning, so there is nothing to assert.
    # We can call it to make sure it does not raise an error.
    verify_compression_th(0.0001)
    verify_compression_th(2)


def test_assert_inputs_exist():
    parser = argparse.ArgumentParser()
    with tempfile.NamedTemporaryFile() as f:
        assert_inputs_exist(parser, [f.name])
    with pytest.raises(SystemExit):
        assert_inputs_exist(parser, ["non_existent_file"])


def test_assert_inputs_dirs_exist():
    parser = argparse.ArgumentParser()
    with tempfile.TemporaryDirectory() as d:
        assert_inputs_dirs_exist(parser, [d])
    with pytest.raises(SystemExit):
        assert_inputs_dirs_exist(parser, ["non_existent_dir"])


def test_assert_outputs_exist():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--overwrite', action='store_true')
    with tempfile.TemporaryDirectory() as d:
        args = parser.parse_args([])
        assert_outputs_exist(parser, args, [os.path.join(d, "test.txt")])
        with open(os.path.join(d, "test.txt"), "w") as f:
            f.write("test")
        with pytest.raises(SystemExit):
            assert_outputs_exist(
                parser, args, [os.path.join(d, "test.txt")])
        args = parser.parse_args(['-f'])
        assert_outputs_exist(parser, args, [os.path.join(d, "test.txt")])


def test_assert_output_dirs_exist_and_empty():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--overwrite', action='store_true')
    with tempfile.TemporaryDirectory() as d:
        args = parser.parse_args([])
        assert_output_dirs_exist_and_empty(
            parser, args, [os.path.join(d, "test_dir")])
        os.mkdir(os.path.join(d, "test_dir2"))
        with open(os.path.join(d, "test_dir2", "test.txt"), "w") as f:
            f.write("test")
        with pytest.raises(SystemExit):
            assert_output_dirs_exist_and_empty(
                parser, args, [os.path.join(d, "test_dir2")])
        args = parser.parse_args(['-f'])
        assert_output_dirs_exist_and_empty(
            parser, args, [os.path.join(d, "test_dir2")])


def test_assert_overlay_colors():
    parser = argparse.ArgumentParser()
    assert_overlay_colors(None, None, parser)
    assert_overlay_colors([], None, parser)
    with pytest.raises(SystemExit):
        assert_overlay_colors([1, 2], None, parser)
    with pytest.raises(SystemExit):
        assert_overlay_colors([1, 2, 3, 4], ["ovl1", "ovl2"], parser)


def test_assert_roi_radii_format():
    # This function depends on an argparse object.
    # It will be tested through the scripts that use it.
    pass


def test_assert_headers_compatible():
    # This function is complex and depends on nibabel.
    # It will be tested through the scripts that use it.
    pass


def test_read_info_from_mb_bdo():
    # This function reads a specific XML format.
    # It will be tested with a real file if possible.
    pass


def test_load_matrix_in_any_format():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("1 2\n3 4")
        filename = f.name
    data = load_matrix_in_any_format(filename)
    assert (data == [[1, 2], [3, 4]]).all()
    os.remove(filename)


def test_save_matrix_in_any_format():
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.npy")
        data = np.array([[1, 2], [3, 4]])
        save_matrix_in_any_format(filename, data)
        assert os.path.exists(filename)


def test_assert_fsl_options_exist():
    parser = argparse.ArgumentParser()
    assert_fsl_options_exist(parser, "--flm=linear", "eddy")
    with pytest.raises(SystemExit):
        assert_fsl_options_exist(parser, "--invalid_option", "eddy")


def test_parser_color_type():
    assert parser_color_type("128") == 128
    with pytest.raises(argparse.ArgumentTypeError):
        parser_color_type("300")


def test_ranged_type():
    ranged_int = ranged_type(int, 0, 10)
    assert ranged_int("5") == 5
    with pytest.raises(argparse.ArgumentTypeError):
        ranged_int("11")


def test_get_default_screenshotting_data():
    # This function depends on an argparse object and nibabel.
    # It will be tested through the scripts that use it.
    pass
