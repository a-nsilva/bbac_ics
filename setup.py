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
            'experiment_evaluator_node = bbac_ics_core.nodes.experiment_evaluator_node:main', 
            'ablation_study = bbac_ics_core.experiments.ablation_study:main',
            'adaptive_eval = bbac_ics_core.experiments.adaptive_eval:main',
            'dynamic_rules_test = bbac_ics_core.experiments.dynamic_rules_test:main',
        ],
    },
    maintainer='Alexandre do Nascimento Silva',
    maintainer_email='alnsilva@uesc.be',
    description='BBAC ICS Framework - Behavioral Access Control for ICS',
    license='Apache 2.0',
    zip_safe=True,
)
