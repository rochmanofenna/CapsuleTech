"""Core BEF zero-knowledge building blocks packaged for import."""

from importlib import import_module

__all__ = [
    "air",
    "fri",
    "stc",
    "zk_geom",
]

# expose subpackages for convenient access
for _name in __all__:
    import_module(f"bef_zk.{_name}")
