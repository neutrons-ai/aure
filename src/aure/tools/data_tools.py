"""
Data loading tools for reflectivity analysis.

Supports various file formats:
- .txt, .dat: Column-based ASCII files
- .refl: NIST reflectivity format
- .ort: ORSO text format

All tools are decorated as LangChain tools for use in the agent workflow.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union
from pathlib import Path
import re


def detect_file_format(file_path: str) -> str:
    """
    Detect the format of a reflectivity data file.
    
    Args:
        file_path: Path to data file
    
    Returns:
        Format string: 'ascii', 'refl', 'ort', or 'unknown'
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix in ['.refl']:
        return 'refl'
    elif suffix in ['.ort']:
        return 'ort'
    elif suffix in ['.txt', '.dat', '.csv', '.asc']:
        return 'ascii'
    else:
        # Try to detect from content
        try:
            with open(file_path, 'r') as f:
                first_lines = [f.readline() for _ in range(10)]
            
            # Check for ORSO header
            if any('# data_source' in line for line in first_lines):
                return 'ort'
            
            # Default to ASCII
            return 'ascii'
        except Exception:
            return 'unknown'


def parse_ascii_columns(file_path: str) -> Dict[str, np.ndarray]:
    """
    Parse ASCII file and detect column structure.
    
    Handles various column formats:
    - Q R
    - Q R dR
    - Q R dR dQ
    
    Args:
        file_path: Path to data file
    
    Returns:
        Dictionary with Q, R, dR, and optionally dQ arrays
    """
    # Read file, skipping comments
    data_lines = []
    header_info = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
            
            # Parse header comments
            if stripped.startswith('#') or stripped.startswith('%'):
                # Look for column hints
                if 'Q' in stripped and ('R' in stripped or 'ref' in stripped.lower()):
                    header_info['has_header'] = True
                continue
            
            # Try to parse as numbers
            try:
                values = [float(x) for x in stripped.split()]
                if len(values) >= 2:
                    data_lines.append(values)
            except ValueError:
                # Skip lines that can't be parsed
                continue
    
    if not data_lines:
        raise ValueError(f"No valid data found in {file_path}")
    
    # Convert to numpy array
    data = np.array(data_lines)
    n_cols = data.shape[1]
    
    # Assign columns
    result = {
        'Q': data[:, 0],
        'R': data[:, 1],
    }
    
    if n_cols >= 3:
        result['dR'] = data[:, 2]
    else:
        # Estimate error as 5% of R (minimum practical estimate)
        result['dR'] = 0.05 * result['R']
    
    if n_cols >= 4:
        result['dQ'] = data[:, 3]
    
    return result


def parse_ort_file(file_path: str) -> Dict[str, np.ndarray]:
    """
    Parse ORSO .ort format file.
    
    Args:
        file_path: Path to .ort file
    
    Returns:
        Dictionary with Q, R, dR arrays and metadata
    """
    # ORSO format has YAML-style header then data columns
    header_lines = []
    data_lines = []
    in_header = True
    
    with open(file_path, 'r') as f:
        for line in f:
            if in_header:
                if line.startswith('#'):
                    header_lines.append(line[1:].strip())
                else:
                    in_header = False
            
            if not in_header:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    try:
                        values = [float(x) for x in stripped.split()]
                        if len(values) >= 2:
                            data_lines.append(values)
                    except ValueError:
                        continue
    
    data = np.array(data_lines)
    
    # ORT format is typically: Q, R, dR, dQ
    result = {
        'Q': data[:, 0],
        'R': data[:, 1],
    }
    
    if data.shape[1] >= 3:
        result['dR'] = data[:, 2]
    else:
        result['dR'] = 0.05 * result['R']
    
    if data.shape[1] >= 4:
        result['dQ'] = data[:, 3]
    
    return result


def load_reflectivity_data(
    file_path: str,
    q_min: Optional[float] = None,
    q_max: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Load reflectivity data from file.
    
    Automatically detects format and parses columns.
    
    Args:
        file_path: Path to data file
        q_min: Minimum Q to include (optional)
        q_max: Maximum Q to include (optional)
    
    Returns:
        Dictionary with Q, R, dR arrays
    """
    # Detect format
    fmt = detect_file_format(file_path)
    
    # Parse file
    if fmt == 'ort':
        data = parse_ort_file(file_path)
    elif fmt in ['ascii', 'refl']:
        data = parse_ascii_columns(file_path)
    else:
        raise ValueError(f"Unknown file format for {file_path}")
    
    # Apply Q range filter
    if q_min is not None or q_max is not None:
        mask = np.ones(len(data['Q']), dtype=bool)
        if q_min is not None:
            mask &= data['Q'] >= q_min
        if q_max is not None:
            mask &= data['Q'] <= q_max
        
        data = {key: arr[mask] for key, arr in data.items()}
    
    return data


def validate_reflectivity_data(
    Q: np.ndarray,
    R: np.ndarray,
    dR: np.ndarray,
) -> Dict[str, Union[bool, List[str]]]:
    """
    Validate reflectivity data for common issues.
    
    Checks:
    - R values in valid range
    - Q spacing and range
    - Error bars reasonable
    - No NaN/inf values
    
    Args:
        Q: Q values
        R: Reflectivity values
        dR: Error values
    
    Returns:
        Dictionary with 'valid' bool and 'issues' list
    """
    issues = []
    warnings = []
    
    # Check for NaN/inf
    if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):
        issues.append("Q contains NaN or inf values")
    if np.any(np.isnan(R)) or np.any(np.isinf(R)):
        issues.append("R contains NaN or inf values")
    if np.any(np.isnan(dR)) or np.any(np.isinf(dR)):
        issues.append("dR contains NaN or inf values")
    
    # Check R range
    if np.any(R > 1.1):
        issues.append(f"R > 1 detected (max={R.max():.3f}): data may not be normalized")
    if np.any(R < 0):
        issues.append("Negative R values detected")
    if np.any(R < 1e-12):
        warnings.append("Very small R values (<1e-12): may cause numerical issues")
    
    # Check Q range
    if Q.min() > 0.02:
        warnings.append(f"Low Q cutoff at {Q.min():.4f}: may miss critical edge")
    if Q.max() < 0.15:
        warnings.append(f"High Q cutoff at {Q.max():.4f}: limited thickness resolution")
    
    # Check Q spacing
    dq = np.diff(Q)
    if np.any(dq <= 0):
        issues.append("Q values not monotonically increasing")
    
    # Check error bars
    if np.any(dR <= 0):
        issues.append("Non-positive error bars detected")
    
    relative_errors = dR / (R + 1e-30)
    if np.median(relative_errors) > 0.5:
        warnings.append("Large relative errors (>50%): data quality may be poor")
    
    # Check for sufficient points
    if len(Q) < 20:
        warnings.append(f"Only {len(Q)} data points: may be insufficient for fitting")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'n_points': len(Q),
        'q_range': (float(Q.min()), float(Q.max())),
        'r_range': (float(R.min()), float(R.max())),
    }


def normalize_reflectivity(
    Q: np.ndarray,
    R: np.ndarray,
    dR: np.ndarray,
    method: str = 'auto',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize reflectivity data to have R=1 at low Q.
    
    Args:
        Q: Q values
        R: Reflectivity values
        dR: Error values
        method: 'auto', 'low_q', or 'max'
    
    Returns:
        Normalized (R, dR) arrays
    """
    if method == 'auto':
        # Use low-Q average if available
        low_q_mask = Q < 0.015
        if np.sum(low_q_mask) >= 3:
            method = 'low_q'
        else:
            method = 'max'
    
    if method == 'low_q':
        low_q_mask = Q < 0.015
        norm_factor = np.mean(R[low_q_mask])
    elif method == 'max':
        norm_factor = R.max()
    else:
        norm_factor = 1.0
    
    if norm_factor > 0:
        R_norm = R / norm_factor
        dR_norm = dR / norm_factor
    else:
        R_norm = R
        dR_norm = dR
    
    return R_norm, dR_norm


def resample_to_log_q(
    Q: np.ndarray,
    R: np.ndarray,
    dR: np.ndarray,
    n_points: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample data to logarithmically spaced Q points.
    
    Useful for FFT analysis of Kiessig fringes.
    
    Args:
        Q: Q values
        R: Reflectivity values
        dR: Error values
        n_points: Number of output points
    
    Returns:
        Resampled (Q, R, dR) arrays
    """
    Q_new = np.logspace(np.log10(Q.min()), np.log10(Q.max()), n_points)
    R_new = np.interp(Q_new, Q, R)
    dR_new = np.interp(Q_new, Q, dR)
    
    return Q_new, R_new, dR_new


def resample_to_linear_q(
    Q: np.ndarray,
    R: np.ndarray,
    dR: np.ndarray,
    n_points: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample data to linearly spaced Q points.
    
    Required for proper FFT analysis.
    
    Args:
        Q: Q values
        R: Reflectivity values
        dR: Error values
        n_points: Number of output points
    
    Returns:
        Resampled (Q, R, dR) arrays
    """
    Q_new = np.linspace(Q.min(), Q.max(), n_points)
    R_new = np.interp(Q_new, Q, R)
    dR_new = np.interp(Q_new, Q, dR)
    
    return Q_new, R_new, dR_new


if __name__ == '__main__':
    # Test data loading
    print("Data loading tools ready.")
    print("Supported formats: .txt, .dat, .refl, .ort")
