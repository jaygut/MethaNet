"""Functional gene quantification module.

This module provides HMM-based functional gene quantification for
methanogenesis and methanotrophy marker genes (mcrA, pmoA, etc.).
"""

from methanet.functional.quantify import FunctionalProfile, FunctionalQuantifier

__all__ = ["FunctionalQuantifier", "FunctionalProfile"]
