#!/usr/bin/env python3

from setuptools import setup, find_packages

package_name = 'bbac_ics_core'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'setuptools',
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'pyyaml'
    ],
    entry_points={
        'console_scripts': [
            'bbac_main_node = bbac_ics_core.nodes.bbac_main_node:main',
            'baseline_manager_node = bbac_ics_core.nodes.baseline_manager_node:main',
            'evaluator_node = bbac_ics_core.nodes.evaluator_node:main',
        ],
    },
    maintainer='Alexandre do Nascimento Silva',
    maintainer_email='alnsilva@uesc.be',
    description='BBAC ICS Framework - Behavioral Access Control for ICS',
    license='Apache 2.0',
    zip_safe=True,
)
