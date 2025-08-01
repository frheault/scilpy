import os
import tempfile
from scilpy.io.varian_fdf import (
    load_fdf,
    add_gradient_info,
    read_file,
    read_directory,
    format_raw_header,
    save_babel,
    write_gradient_information,
    correct_procpar_intensity,
    get_gain,
)


def test_load_fdf():
    # TODO: Implement this test
    pass


def test_add_gradient_info():
    # TODO: Implement this test
    pass


def test_read_file():
    # TODO: Implement this test
    pass


def test_read_directory():
    # TODO: Implement this test
    pass


def test_format_raw_header():
    # TODO: Implement this test
    pass


def test_save_babel():
    # TODO: Implement this test
    pass


def test_write_gradient_information():
    # TODO: Implement this test
    pass


def test_correct_procpar_intensity():
    # TODO: Implement this test
    pass


def test_get_gain():
    with tempfile.TemporaryDirectory() as d:
        procpar_path = os.path.join(d, "procpar")
        with open(procpar_path, "w") as f:
            f.write("gain\n")
            f.write("1 1.23\n")
        assert get_gain(d) == 1.23
