#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_enforce_streamlines_similarity_from_endpoints.py', '--help')
    assert ret.success
