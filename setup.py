#!/usr/bin/env python
# encoding: utf-8

import os
import sys

from setuptools import setup

try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/csalgs")
    from csalgs import __version__ as version
except:
    version = "unknown"


if __name__ == '__main__':
    setup(
        name='csalgs',
        author='Daniel Suess',
        version=version,
        packages=['csalgs'],
        package_dir={'csalgs': 'csalgs'},
        license="GPL",
        install_requires=['numpy', 'cvxpy', 'mpnum>=0.2'],
        keywords=[],
        classifiers=[
            "Operating System :: OS Indendent",
            "Programming Language :: Python :: 3.3",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Intended Audience :: Science/Research"
        ],
        platforms=['ALL'],
    )
