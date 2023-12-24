#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "torch~=2.1.0",
    "numpy~=1.26.1",
    "scikit-learn~=1.3.1",
    "scipy~=1.11.3",
    "tqdm~=4.66.1",
    "wandb==0.15.12",
    "torchdrug",
    "rdkit",
    "pandas",
    "ml-collections",
    "molfeat",
    "torch_geometric",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Leon Gerard",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    description="A python package to perform molecular property prediction with pre-trained regression models.",
    entry_points={
        "console_scripts": [
            "molproperty_prediction=molproperty_prediction.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords="molproperty_prediction",
    name="molproperty_prediction",
    packages=find_packages(
        include=["molproperty_prediction", "molproperty_prediction.*"]
    ),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/LeonGerBerlin/molproperty_prediction",
    version="0.1.0",
    zip_safe=False,
)
