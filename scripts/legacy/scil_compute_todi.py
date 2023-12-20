#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_tractogram_compute_TODI import main as new_main


DEPRECATION_MSG = """
This script has been renamed scil_tractogram_compute_TODI.py. Please change
your existing pipelines accordingly.

"""


@deprecate_script("scil_compute_todi", DEPRECATION_MSG, '1.7.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
