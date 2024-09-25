#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_tractogram_resample import main as new_main


DEPRECATION_MSG = """
This script has been renamed scil_tractogram_resample.py. Please change
your existing pipelines accordingly.

"""


@deprecate_script("scil_resample_tractogram.py", DEPRECATION_MSG, '2.0.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
