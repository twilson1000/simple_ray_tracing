#!/usr/bin/env python3
# -*- coding: utf-8
'''

'''
# Standard imports.
from setuptools import setup, find_packages

# Custom imports.


__author__ = "Thomas Wilson"
__version__ = "0.1"
__url__ = "https://github.com/twilson1000/simple_ray_tracing"

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name="simple_ray_tracing",
    version=__version__,
    description="Microwave ray tracing code for simple plasmas",
    long_description=readme(),
    url=__url__,
    packages=["simple_ray_tracing"],
    author=__author__,
    author_email="thomas.wilson@ukaea.uk",
    python_requires=">=3.9",
    install_requires=[
        "matplotlib",
        "numpy >=1.26.4",
        "scipy >=1.13",
        "netCDF4"
    ]
)
