"""Cluster import shims for Hugging Face helper libraries.

TRL 1.8 calls ``importlib.metadata.packages_distributions()`` during import.
On Expanse, with the Pixi env on Lustre, that global metadata walk can hang for
minutes before training even starts. TRL only needs a few package-name mappings
for the optional-dependency checks used here, so provide those directly.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os


def patch_importlib_metadata_for_trl() -> None:
    """Avoid TRL's slow global package metadata scan on Lustre filesystems."""
    if os.environ.get("BIOREASON_DISABLE_TRL_IMPORT_SHIM"):
        return
    if getattr(importlib.metadata, "_bioreason_trl_patch", False):
        return

    original = importlib.metadata.packages_distributions

    def fast_packages_distributions():
        mapping = {}
        for import_name, dist_name in {
            "transformers": "transformers",
            "peft": "peft",
            "trl": "trl",
            "datasets": "datasets",
            "accelerate": "accelerate",
            "torch": "torch",
            "requests": "requests",
        }.items():
            if importlib.util.find_spec(import_name) is not None:
                mapping[import_name] = [dist_name]
        return mapping

    importlib.metadata._bioreason_original_packages_distributions = original
    importlib.metadata.packages_distributions = fast_packages_distributions
    importlib.metadata._bioreason_trl_patch = True
