"""
Material database for neutron reflectivity analysis.

Uses the `periodictable` package for accurate scattering length density (SLD)
calculations based on chemical formulas and densities.

Key functions:
- compute_sld(formula, density): Calculate SLD for any chemical formula
- lookup_material(name): Find common materials by name/alias
- get_contrast_match_ratio(target_sld): Calculate D2O/H2O mixture ratio

SLD values are in units of 10⁻⁶ Å⁻²
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import periodictable
from periodictable import formula as pt_formula, neutron_sld


@dataclass
class Material:
    """
    Material with SLD information.
    
    Can be defined either by:
    1. Chemical formula + density (calculated SLD)
    2. Fixed SLD value (for mixtures, polymers, etc.)
    """
    name: str
    formula: Optional[str] = None  # Chemical formula (e.g., 'SiO2')
    density: Optional[float] = None  # g/cm³
    sld: Optional[Union[float, Tuple[float, float]]] = None  # Override or range
    aliases: List[str] = field(default_factory=list)
    category: str = 'other'
    notes: Optional[str] = None
    
    def get_sld(self) -> float:
        """Calculate or return SLD value."""
        if self.sld is not None:
            if isinstance(self.sld, tuple):
                return (self.sld[0] + self.sld[1]) / 2
            return self.sld
        
        if self.formula and self.density:
            return compute_sld(self.formula, self.density)
        
        raise ValueError(f"Cannot compute SLD for {self.name}: missing formula/density or fixed SLD")
    
    def get_sld_range(self) -> Tuple[float, float]:
        """Get SLD range (for materials with variable SLD)."""
        if isinstance(self.sld, tuple):
            return self.sld
        sld_val = self.get_sld()
        return (sld_val, sld_val)


def compute_sld(chemical_formula: str, density: float) -> float:
    """
    Compute neutron SLD for a chemical formula at given density.
    
    Uses periodictable for accurate calculation based on
    neutron scattering lengths of constituent atoms.
    
    Args:
        chemical_formula: Chemical formula (e.g., 'SiO2', 'C8H8', 'D2O')
        density: Mass density in g/cm³
    
    Returns:
        SLD in units of 10⁻⁶ Å⁻²
    
    Example:
        >>> compute_sld('Si', 2.33)
        2.073
        >>> compute_sld('D2O', 1.107)
        6.393
    """
    try:
        f = pt_formula(chemical_formula)
        # neutron_sld returns (real, imag, incoh) in units of 10^-6 Å^-2
        sld_real, sld_imag, sld_incoh = neutron_sld(f, density=density)
        return sld_real
    except Exception as e:
        raise ValueError(f"Cannot compute SLD for '{chemical_formula}': {e}")


def compute_sld_with_absorption(chemical_formula: str, density: float) -> Tuple[float, float, float]:
    """
    Compute full neutron SLD including absorption and incoherent components.
    
    Args:
        chemical_formula: Chemical formula
        density: Mass density in g/cm³
    
    Returns:
        Tuple of (real_SLD, imaginary_SLD, incoherent_SLD) in 10⁻⁶ Å⁻²
    """
    f = pt_formula(chemical_formula)
    return neutron_sld(f, density=density)


# ============================================================================
# COMPOUND DENSITIES (g/cm³)
# periodictable has natural densities for elements, but not for compounds
# ============================================================================

DENSITIES: Dict[str, float] = {
    # Solvents
    'D2O': 1.107,
    'H2O': 1.00,
    'C7H8': 0.867,   # toluene
    'C7D8': 0.943,   # d-toluene
    'C2H6O': 0.789,  # ethanol
    'C2D6O': 0.888,  # d-ethanol
    'C6H14': 0.659,  # hexane
    'C4H8O': 0.889,  # THF (tetrahydrofuran)
    'C4D8O': 0.985,  # d-THF
    # Oxides
    'SiO2': 2.20,
    'Al2O3': 3.98,
    'TiO2': 4.23,
    # Polymers (monomer unit)
    'C8H8': 1.05,    # polystyrene
    'C8D8': 1.12,    # d-polystyrene
    'C5H8O2': 1.18,  # PMMA
    'C5D8O2': 1.25,  # d-PMMA
    'C2H4O': 1.21,   # PEO
    'C2H6OSi': 0.97, # PDMS
}

# ============================================================================
# ALIASES: Map common names to chemical formulas
# ============================================================================

ALIASES: Dict[str, str] = {
    # Substrates
    'silicon': 'Si', 'si': 'Si', 'silicon wafer': 'Si',
    'quartz': 'SiO2', 'fused silica': 'SiO2', 'silica': 'SiO2',
    'sapphire': 'Al2O3', 'alumina': 'Al2O3', 'corundum': 'Al2O3',
    'germanium': 'Ge', 'ge': 'Ge',
    # Solvents
    'heavy water': 'D2O', 'deuterated water': 'D2O', 'd2o': 'D2O',
    'water': 'H2O', 'light water': 'H2O', 'h2o': 'H2O',
    'toluene': 'C7H8', 'd-toluene': 'C7D8', 'd8-toluene': 'C7D8',
    'ethanol': 'C2H6O', 'EtOH': 'C2H6O',
    'd-ethanol': 'C2D6O', 'd6-ethanol': 'C2D6O',
    'hexane': 'C6H14', 'n-hexane': 'C6H14',
    'THF': 'C4H8O', 'thf': 'C4H8O', 'tetrahydrofuran': 'C4H8O',
    'd-THF': 'C4D8O', 'd8-THF': 'C4D8O', 'dTHF': 'C4D8O',
    # Metals
    'gold': 'Au', 'titanium': 'Ti', 'chromium': 'Cr', 'chrome': 'Cr',
    'nickel': 'Ni', 'copper': 'Cu', 'aluminum': 'Al', 'aluminium': 'Al',
    'iron': 'Fe', 'platinum': 'Pt',
    # Oxides
    'native oxide': 'SiO2', 'thermal oxide': 'SiO2', 'silicon dioxide': 'SiO2',
    'titania': 'TiO2', 'titanium dioxide': 'TiO2',
    'aluminum oxide': 'Al2O3',
    # Polymers
    'polystyrene': 'C8H8', 'PS': 'C8H8', 'h-PS': 'C8H8',
    'd-polystyrene': 'C8D8', 'd-PS': 'C8D8', 'd8-PS': 'C8D8', 'dPS': 'C8D8',
    'PMMA': 'C5H8O2', 'pmma': 'C5H8O2', 'acrylic': 'C5H8O2',
    'd-PMMA': 'C5D8O2', 'd8-PMMA': 'C5D8O2',
    'PEO': 'C2H4O', 'peo': 'C2H4O', 'PEG': 'C2H4O',
    'PDMS': 'C2H6OSi', 'pdms': 'C2H6OSi', 'silicone': 'C2H6OSi',
}


def resolve_formula(name: str) -> Optional[str]:
    """
    Resolve a material name to its chemical formula.
    
    Args:
        name: Material name or alias
    
    Returns:
        Chemical formula if found, None otherwise
    """
    # Check if it's already a formula (try parsing it)
    try:
        pt_formula(name)
        return name
    except:
        pass
    
    # Look up alias
    name_lower = name.lower().strip()
    if name_lower in ALIASES:
        return ALIASES[name_lower]
    if name in ALIASES:
        return ALIASES[name]
    
    return None


def get_density(formula: str) -> Optional[float]:
    """
    Get density for a chemical formula.
    
    Uses DENSITIES lookup table for compounds, or periodictable
    for elements with natural density.
    
    Args:
        formula: Chemical formula
    
    Returns:
        Density in g/cm³, or None if unknown
    """
    # Check our density table
    if formula in DENSITIES:
        return DENSITIES[formula]
    
    # Try periodictable's natural density (works for elements)
    try:
        f = pt_formula(formula)
        if hasattr(f, 'density') and f.density is not None:
            return f.density
    except:
        pass
    
    return None


def lookup_material(name: str) -> Optional[Material]:
    """
    Find material by name or alias and return a Material object.
    
    Args:
        name: Material name to look up
    
    Returns:
        Material if found, None otherwise
    """
    # Handle special cases
    if name.lower() in ('air', 'vacuum', 'atmosphere'):
        return Material(name='air', sld=0.0)
    
    formula = resolve_formula(name)
    if formula is None:
        return None
    
    density = get_density(formula)
    if density is None:
        return None
    
    return Material(name=name, formula=formula, density=density)


def get_sld(name_or_formula: str, density: Optional[float] = None) -> float:
    """
    Get SLD for a material by name or chemical formula.
    
    Args:
        name_or_formula: Material name (e.g., 'silicon') or formula (e.g., 'Si')
        density: Density in g/cm³ (optional, looked up if not provided)
    
    Returns:
        SLD in 10⁻⁶ Å⁻²
    
    Examples:
        >>> get_sld('silicon')
        2.073
        >>> get_sld('SiO2', density=2.2)
        3.475
        >>> get_sld('D2O')
        6.393
    """
    # Handle special cases
    if name_or_formula.lower() in ('air', 'vacuum', 'atmosphere'):
        return 0.0
    
    # Resolve to formula
    formula = resolve_formula(name_or_formula)
    if formula is None:
        formula = name_or_formula  # Assume it's already a formula
    
    # Get density
    if density is None:
        density = get_density(formula)
    
    if density is None:
        raise ValueError(
            f"Unknown density for '{name_or_formula}'. "
            f"Provide density explicitly: get_sld('{formula}', density=X.XX)"
        )
    
    return compute_sld(formula, density)


def get_contrast_match_ratio(
    target_sld: float,
    protiated_solvent: str = 'H2O',
    deuterated_solvent: str = 'D2O',
) -> float:
    """
    Calculate deuterated/protiated solvent volume fraction to match target SLD.
    
    Args:
        target_sld: Target SLD to match (10⁻⁶ Å⁻²)
        protiated_solvent: Name/formula of protiated solvent (default: 'H2O')
        deuterated_solvent: Name/formula of deuterated solvent (default: 'D2O')
    
    Returns:
        Volume fraction of deuterated solvent (0 to 1)
    
    Examples:
        >>> get_contrast_match_ratio(2.07)  # Match silicon with D2O/H2O
        0.38  # ~38% D2O
        
        >>> get_contrast_match_ratio(2.07, 'toluene', 'd-toluene')  # Using toluene
        0.31  # ~31% d-toluene
    """
    # Get SLDs for solvents
    sld_deuterated = get_sld(deuterated_solvent)
    sld_protiated = get_sld(protiated_solvent)
    
    # Linear interpolation
    fraction = (target_sld - sld_protiated) / (sld_deuterated - sld_protiated)
    return max(0.0, min(1.0, fraction))


def get_mixture_sld(
    fraction_deuterated: float,
    protiated_solvent: str = 'H2O',
    deuterated_solvent: str = 'D2O',
) -> float:
    """
    Calculate SLD of a deuterated/protiated solvent mixture.
    
    Args:
        fraction_deuterated: Volume fraction of deuterated solvent (0 to 1)
        protiated_solvent: Name/formula of protiated solvent (default: 'H2O')
        deuterated_solvent: Name/formula of deuterated solvent (default: 'D2O')
    
    Returns:
        SLD of mixture in 10⁻⁶ Å⁻²
    
    Examples:
        >>> get_mixture_sld(0.5)  # 50% D2O / 50% H2O
        2.91
        
        >>> get_mixture_sld(0.5, 'toluene', 'd-toluene')  # 50% d-toluene
        2.94
    """
    sld_deuterated = get_sld(deuterated_solvent)
    sld_protiated = get_sld(protiated_solvent)
    return sld_protiated + fraction_deuterated * (sld_deuterated - sld_protiated)


if __name__ == '__main__':
    print("Testing periodictable-based SLD calculations...\n")
    
    # Test common materials by name
    test_materials = ['silicon', 'D2O', 'gold', 'polystyrene', 'd-PS', 'native oxide']
    print("Common materials (by name/alias):")
    for name in test_materials:
        sld = get_sld(name)
        print(f"  {name}: SLD = {sld:.3f} × 10⁻⁶ Å⁻²")
    
    # Test direct formula calculation
    print("\nDirect formula calculations:")
    formulas = [
        ('Si', 2.33),
        ('SiO2', 2.20),
        ('D2O', 1.107),
        ('C8H8', 1.05),  # Polystyrene
        ('C8D8', 1.12),  # d-Polystyrene
        ('Au', 19.3),
        ('Fe3O4', 5.17),  # Magnetite - arbitrary formula
    ]
    for formula, density in formulas:
        sld = compute_sld(formula, density)
        print(f"  {formula} (ρ={density}): SLD = {sld:.3f} × 10⁻⁶ Å⁻²")
    
    # Test contrast matching
    print("\nContrast match D2O fractions:")
    targets = [('Silicon', 2.07), ('Quartz', 4.18), ('Sapphire', 5.72)]
    for name, sld in targets:
        fraction = get_contrast_match_ratio(sld)
        print(f"  {name} (SLD={sld}): {fraction*100:.1f}% D2O")
