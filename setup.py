
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

import subprocess, sys

def julia_instantiate():
    args = ["julia", "pomato/_installation/julia_instantiate.jl"]
    with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT) as programm:
        for line in programm.stdout:
            print(line.decode(errors="ignore").strip())

def check_for_julia_and_gurobi():
    # Check of jula exists
    import distutils.spawn
    import os
    julia_exists = bool(distutils.spawn.find_executable("julia"))
    gurobi_exists = bool("GUROBI_HOME" in os.environ)

    if not julia_exists:
        raise ImportError('Julia Executable not found on path.')
    else:
        print("Julia Found!")
    if not gurobi_exists:
        raise ImportError('GUROBI_HOME not found on path.')
    else:
        print("Gurobi Found!")

        
# https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
class DevelopCommand(develop):
    """Pre-installation for development mode."""
    def run(self):
        check_for_julia_and_gurobi()
        develop.run(self)
        julia_instantiate()

class InstallCommand(install):
    """Pre-installation for installation mode."""
    def run(self):
        check_for_julia_and_gurobi()
        install.run(self)
        julia_instantiate()

setup(name='pomato',
      version='0.0.1',
      description='Power Market Tool',
      author='Richard Weinhold',
      author_email='riw@wip.tu-berlin.de',
      url='https://github.com/korpuskel91/pomato',
      packages=find_packages(),
      python_requires='>=3.6',
      include_package_data = True,
      install_requires=[
        'alabaster',
        'bokeh',
        'jinja2',
        'matplotlib',
        'numpy',
        'numpydoc',
        'openpyxl',
        'pandas',
        'pathlib',
        'psutil',
        'pyproj',
        'scikit-learn',
        'scipy',
        'xlrd'],
      cmdclass={
        'develop': DevelopCommand,
        'install': InstallCommand,}
     )

