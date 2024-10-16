#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bst.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_screenshot(script_runner):
    in_fa = os.path.join(get_home(), 'bst',
                         'fa.nii.gz')

    ret = script_runner.run(
        "scil_screenshot_volume.py", in_fa, 'fa.png'
    )
    assert ret.success


def test_help_option(script_runner):

    ret = script_runner.run(
        "scil_screenshot_volume.py", "--help"
    )
    assert ret.success
