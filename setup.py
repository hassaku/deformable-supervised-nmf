#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import with_statement
from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="deformable-supervised-nmf",
    version="0.1.0",
    description="Implementation for deformable supervised nmf",
    long_description="",
    author="hassaku",
    author_email="hassaku.apps@gmail.com",
    url="https://github.com/hassaku",
    py_modules=["deformable-supervised-nmf"],
    include_package_data=True,
    install_requires=[],
    tests_require=["nose"],
    license="MIT",
    keywords="",
    zip_safe=False,
    classifiers=[]
)
