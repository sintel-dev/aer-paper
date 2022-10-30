# -*- coding: utf-8 -*-

__author__ = """Dongyu Liu"""
__email__ = 'dongyu@mit.edu'
__version__ = '0.0.1.dev0'

import os

_BASE_PATH = os.path.abspath(os.path.dirname(__file__))
MLBLOCKS_PRIMITIVES = os.path.join(_BASE_PATH, 'primitives', 'jsons')
MLBLOCKS_PIPELINES = tuple(
    dirname
    for dirname, _, _ in os.walk(os.path.join(_BASE_PATH, 'pipelines'))
)