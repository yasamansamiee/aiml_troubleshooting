# -*- coding: utf-8 -*-
r"""
Script to minify the package upon install.
"""

__author__ = "Stefan Ulbrich"
__copyright__ = "Copyright 2018-2020, Acceptto Corporation, All rights reserved."
__credits__ = []
__license__ = "Acceptto Confidential"
__version__ = ""
__maintainer__ = "Stefan Ulbrich"
__email__ = "Stefan.Ulbrich@acceptto.com"
__status__ = "alpha"
__date__ = "2020-03-12"

from setuptools import setup
from setuptools.command.install import install

from  setuptools.command.build_py import build_py

try:
    from python_minifier import minify
except ImportError:
    raise ImportError("python minifier required for installation")


# https://blog.niteo.co/setuptools-run-custom-code-in-setup-py/
# https://github.com/python-poetry/poetry/issues/265
# https://jichu4n.com/posts/how-to-add-custom-build-steps-and-commands-to-setuppy/
# https://github.com/python-poetry/poetry/issues/11
# https://github.com/sdispater/pendulum/blob/master/build.py
# https://github.com/facebookincubator/bowler

class CustomBuildPyCommand(build_py):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):

        print(f"Stef: {self.packages}, {self.py_modules}")
        super().run()

    def build_module(self, *args, **kwargs):
        (dest_name, copied) = super().build_module(*args, **kwargs)
        print(f"{'written' if copied else 'failed to write'}: {dest_name}")
        if copied:
            with open(dest_name, 'r') as file:
                data = file.read()
            with open(dest_name, 'w') as file:
                file.write(minify(data,remove_literal_statements=True) )
                
        return (dest_name, copied)

# XXX how to add python-minifier as a build requirement
# XXX how to call python-minifier from build process
# Note: build is called when installing the wheel! The wheel is not minified!

def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    print("Called build")
    setup_kwargs.update(
        {"cmdclass": {'build_py': CustomBuildPyCommand}}
    )