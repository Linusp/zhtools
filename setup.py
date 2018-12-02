#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages


VERSION = '0.0.1'
REQS = []


setup(
    name='zhtools',
    version=VERSION,
    description='',
    license='MIT',
    packages=find_packages(),
    install_requires=REQS,
    include_package_data=True,
    zip_safe=False,
)
