#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Repository for sound source localization and tracking using probabilistic models",
    author="Victorletzelter",
    author_email="letzelter.victor@hotmail.fr",
    url="https://github.com/Victorletzelter/lightning-hydra-template.git",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
