"""
This module is created to set our project up
and define their authors.
"""

from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='DL project useful to predict if a product reaches the top standards'
                ' depending on reviews made by clients on e-commerce',
    author='sexism_identifiers (Valèria Caro, Esther Fanyanàs and Claudia Len)',
    license='MIT',
)
