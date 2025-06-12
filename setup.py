from setuptools import setup, find_packages

setup(
    name="learn2trust",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pytest",
    ],
)