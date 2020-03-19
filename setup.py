#!/usr/bin/env python

from distutils.core import setup

setup(name='pomato',
      version='0.2',
      description='Power Market Tool',
      author='Richard Weinhold',
      author_email='riw@wip.tu-berlin.de',
      url='https://github.com/korpuskel91/pomato',
      packages=setuptools.find_packages(),
      python_requires='>=3.6',
      scrips=['project_files/julia_instantiate.py']
     )