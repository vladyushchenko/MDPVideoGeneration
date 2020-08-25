from os import path

from setuptools import find_packages, setup

import mdp_video

with open(path.abspath(path.join(path.dirname(__file__), "README.md")), "r") as fh:
    LONG_DESCRIPTION = fh.read()

_CLASSIFIERS = [
    "Natural Language :: English",
    "Programming Language :: Python :: 3.6",
]

setup(
    name="mdp_video",
    zip_safe=False,
    author="Vladyslav Yushchenko",
    author_email="yushchenko<dot>vladyslav<at>gmail<dot>com",
    version=mdp_video.__version__,
    packages=find_packages(),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    classifiers=_CLASSIFIERS,
)
