"""Material database module using periodictable for accurate SLD calculations."""

from .materials import (
    Material,
    DENSITIES,
    ALIASES,
    compute_sld,
    resolve_formula,
    get_density,
    lookup_material,
    get_sld,
)

__all__ = [
    'Material',
    'DENSITIES',
    'ALIASES',
    'compute_sld',
    'resolve_formula',
    'get_density',
    'lookup_material',
    'get_sld',
]
