#
# Copyright (C) 2020, Vladyslav Yushchenko, iNTENCE automotive electronics GmbH
#

from os import path

from setuptools import find_packages, setup

import mdp_video

NAME = "mdp_video"

with open(path.abspath(path.join(path.dirname(__file__), "README.md")), "r") as fh:
    LONG_DESCRIPTION = fh.read()

# see https://pypi.org/pypi?%3Aaction=list_classifiers
_CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "Environment :: ",
    "Framework :: ",
    "Intended Audience :: ",
    "License :: Other/Proprietary License",
    "Natural Language :: English",
    "Operating System :: ",
    "Programming Language :: Python :: 3.6",
    "Topic :: ",
    "Typing :: Typed",
]

setup(
    name=NAME,
    zip_safe=False,
    author="Vladyslav Yushchenko",
    author_email="yushchenko.intence@gmail.com",
    version=mdp_video.__version__,
    packages=find_packages(),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    classifiers=_CLASSIFIERS,
)
