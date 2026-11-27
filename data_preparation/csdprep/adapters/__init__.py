"""Dataset adapters. Each module exposes load(root, **options) -> csdprep.schema.Dataset."""
import importlib

ADAPTERS = ["nih_cxr", "chexpert", "ham10000", "ptbxl", "abide"]


def load(name, root, **opts):
    return importlib.import_module(f"csdprep.adapters.{name}").load(root, **opts)
