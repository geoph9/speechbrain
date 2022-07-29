#!/usr/bin/env python3
import os
import sys
import site
import setuptools
from distutils.core import setup


# Editable install in user site directory can be allowed with this hack:
# https://github.com/pypa/pip/issues/7953.
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

with open("README.md") as f:
    long_description = f.read()

with open(os.path.join("speechbrain", "version.txt")) as f:
    version = f.read().strip()

with open(os.path.join("speechbrain", "requirements.txt")) as f:
    required = f.read().splitlines()
required = [p for p in required if not p.startswith("-")]

setup(
    name="speechbrain-geoph9",
    version=version,
    description="All-in-one speech toolkit in pure Python and Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mirco Ravanelli & Others",
    author_email="speechbrain@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    packages=setuptools.find_packages(),
    package_data={"speechbrain": ["version.txt", "log-config.yaml", "requirements.txt"]},
    setup_requires=["wheel>=0.37.1"],
    install_requires=required,
    python_requires=">=3.7",
    url="https://speechbrain.github.io/",
)
