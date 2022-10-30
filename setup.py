#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

try:
    with open('README.md', encoding='utf-8') as readme_file:
        readme = readme_file.read()
except IOError:
    readme = ''

try:
    with open('HISTORY.md', encoding='utf-8') as history_file:
        history = history_file.read()
except IOError:
    history = ''


install_requires = [
    'orion-ml@git+https://git@github.com/sintel-dev/Orion.git@aer', # from aer branch
    'tensorflow==2.3.4', # remove later
    'notebook'
]

setup(
    author='Sarah Alnegheimish',
    author_email='smish@mit.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Producing results for AER.",
    entry_points={
        'mlblocks': [
            'primitives=aer:MLBLOCKS_PRIMITIVES',
            'pipelines=aer:MLBLOCKS_PIPELINES'
        ],
    },
    include_package_data=True,
    install_requires=install_requires,
    keywords='aer',
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    name='aer',
    packages=find_packages(include=['aer', 'aer.*']),
    python_requires='>=3.6,<3.8',
    url='https://github.com/dyuliu/aer-paper',
    version='0.0.1.dev0',
    zip_safe=False,
)