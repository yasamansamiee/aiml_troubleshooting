# -*- coding: utf-8 -*-
r"""
Modul imports.
"""

__author__ = "Stefan Ulbrich"
__copyright__ = "Copyright 2018-2020, Acceptto Corporation, All rights reserved."
__license__ = "Acceptto Confidential"
__maintainer__ = "Stefan Ulbrich"
__email__ = "Stefan.Ulbrich@acceptto.com"
__date__ = "2020-03-21"

__all__ = ["KuberspatiotemporalModel", "KuberModel", "SpatialModel"]

# https://stackoverflow.com/a/29509611
from .compound import CompoundModel, Feature
from .kuber import KuberModel
from .spatial import SpatialModel

KuberspatiotemporalModel = CompoundModel
