"""QAMQOR: a reproducible benchmark for engagement recognition in
Robot-Assisted Autism Therapy.

This package provides the shared building blocks used by the benchmark scripts:
central configuration (:mod:`qamqor.config`), evaluation metrics
(:mod:`qamqor.metrics`), temporal feature construction (:mod:`qamqor.features`),
and data loading / split generation (:mod:`qamqor.data`).
"""

from . import config, data, features, metrics  # noqa: F401

__all__ = ["config", "data", "features", "metrics"]
__version__ = "1.0.0"
