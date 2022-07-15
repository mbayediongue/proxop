"""
Setup file for the library: proxop.

Jun 2022
"""

import os
from setuptools import setup


def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)


proj_description = 'This Python library provides an implementation of the proximity ' \
        'operator of many classical functions to solve non-smooth optimization' \
        ' problems.'

# Setup
setup(
    name='proxop',
    description=proj_description,
    long_description=open(src('README.md')).read(),
    long_description_content_type='text/markdown',
    keywords=['optimization'
              'inverse problems',
              'proximity operator',
              'non smooth optimization problem'
              'convex optimization',
              'large-scale optimization'],


    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: CeCILL-B Free Software License Agreement (CECILL-B)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
    ],

    author='mbayediongue',
    author_email='mbayediongue22@gmail.com',
    python_requires='>3.8.0',
    install_requires=['numpy >= 1.17.3', 'scipy >= 1.8.0'],
    use_scm_version=dict(root='.',
                         relative_to=__file__,
                         write_to=src('proxop/version.py')),
    setup_requires=['setuptools_scm'],
    zip_safe=True)
