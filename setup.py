## setup.py
from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='CHALK',
    version='0.1',
    packages=find_packages(),
    # package_dir={'': 'src'},
    # py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
)



# import sys
# import os
# from cx_Freeze import setup, Executable

# # ADD FILES
# files = ['icon.ico','themes/']

# # TARGET
# target = Executable(
#     script="main.py",
#     base="Win32GUI",
#     icon="icon.ico"
# )

# # SETUP CX FREEZE
# setup(
#     name = "PyDracula",
#     version = "1.0",
#     description = "Modern GUI for Python applications",
#     author = "Wanderson M. Pimenta",
#     options = {'build_exe' : {'include_files' : files}},
#     executables = [target]
    
# )
