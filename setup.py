
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

import subprocess, sys, os
from pathlib import Path

def julia_instantiate(install_lib_path):
    args = ["julia", "_installation/julia_install_from_git.jl", "main", "construction"]   
    # raise ImportError("package path %s", package_path)
    package_path = Path(install_lib_path).joinpath("pomato")
    process = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, cwd=str(package_path))
    process.communicate()

def check_for_julia():
    # Check if julia exists
    import distutils.spawn
    import os
    if not bool(distutils.spawn.find_executable("julia")):
        raise ImportError('Julia Executable not found on path.')
    else:
        print("Julia Found!")

# https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
class DevelopCommand(develop):
    """Pre-installation for development mode."""
    def run(self):
        #check_for_julia()
        develop.run(self)

class InstallCommand(install):
    """Pre-installation for installation mode."""
    def run(self):
        check_for_julia()
        install.run(self)
        julia_instantiate(self.install_lib)

setup(name='pomato',
      version='0.5.0',
      description='Power Market Tool',
      author='Richard Weinhold',
      author_email='riw@wip.tu-berlin.de',
      url='https://github.com/richard-weinhold/pomato',
      packages=find_packages(),
      python_requires='>=3.6',
      include_package_data = True,
      install_requires=[
        # 'alabaster',
        # 'numpydoc',
        'matplotlib',
        'numpy',
        'xlrd',
        'openpyxl',
        'pandas',
        'pathlib',
        'psutil',
        'scipy',
        'plotly',
        'progress',
        'geojson',
        'shapely',
        'logaugment',
        'kaleido',
        'dash',
        'dash_bootstrap_components',
        'dash_daq',
        'pycddlib',
        'requests',
        'pyarrow'
        # 'pypoman'
        ],
      cmdclass={
        'develop': DevelopCommand,
        'install': InstallCommand,}
     )

