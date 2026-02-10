"""
Feature extraction tools for reflectivity analysis.

These tools extract physics-meaningful features from R(Q) curves:
- Critical edge detection (Qc -> SLD)
- Oscillation/fringe analysis (period -> thickness)
- Roughness estimation (high-Q decay)
- Layer count estimation

Based on the physics features used in the forward PINN models:
- Q/Qc ratios for critical edge identification
- Q·d products for interference fringes
- SLD contrasts for amplitude modulation
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional
import warnings


def extract_critical_edges(
    Q: np.ndarray,
    R: np.ndarray,
    prominence: float = 0.5,
    min_qc: float = 0.005,
    max_qc: float = 0.05,
) -> List[Dict]:
    """
    Find critical edge(s) and estimate corresponding SLD(s).
    
    The critical edge occurs where R drops sharply from the plateau.
    For total external reflection: Qc = 4*sqrt(π*SLD)
    
    Args:
        Q: Q values (Å⁻¹)
        R: Reflectivity values
        prominence: Peak prominence for edge detection
        min_qc: Minimum Qc to consider
        max_qc: Maximum Qc to consider
    
    Returns:
        List of dicts with {Qc, estimated_SLD, confidence}
    """
    # Work in log space for better gradient calculation
    # Use np.maximum to handle negative R values from background subtraction
    log_R = np.log10(np.maximum(R, 1e-12))
    
    # Smooth the data slightly to reduce noise
    if len(log_R) > 20:
        window = min(11, len(log_R) // 5)
        if window % 2 == 0:
            window += 1
        log_R_smooth = savgol_filter(log_R, window, 3)
    else:
        log_R_smooth = log_R
    
    # Calculate gradient
    dlogR_dQ = np.gradient(log_R_smooth, Q)
    
    # Find minima in gradient (steepest descent = critical edge)
    # Only look in reasonable Qc range
    q_mask = (Q >= min_qc) & (Q <= max_qc)
    
    if not np.any(q_mask):
        return []
    
    # Find peaks in negative gradient (descents)
    neg_gradient = -dlogR_dQ.copy()
    neg_gradient[~q_mask] = 0  # Mask out regions outside Qc range
    
    peaks, properties = find_peaks(neg_gradient, prominence=prominence)
    
    results = []
    for peak_idx in peaks:
        qc = Q[peak_idx]
        
        # Estimate SLD from Qc: SLD = (Qc/4)² / π × 10⁶
        sld = (qc / 4)**2 / np.pi * 1e6
        
        # Confidence based on how sharp the edge is
        edge_sharpness = abs(dlogR_dQ[peak_idx])
        if edge_sharpness > 50:
            confidence = 'high'
        elif edge_sharpness > 20:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        results.append({
            'Qc': float(qc),
            'estimated_SLD': float(sld),
            'confidence': confidence,
            'gradient': float(dlogR_dQ[peak_idx]),
        })
    
    # If no peaks found, look for the steepest point
    if not results:
        idx_max = np.argmin(dlogR_dQ[q_mask])
        idx = np.where(q_mask)[0][idx_max]
        qc = Q[idx]
        sld = (qc / 4)**2 / np.pi * 1e6
        
        results.append({
            'Qc': float(qc),
            'estimated_SLD': float(sld),
            'confidence': 'low',
            'gradient': float(dlogR_dQ[idx]),
        })
    
    # Sort by Qc
    results.sort(key=lambda x: x['Qc'])
    
    return results


def extract_kiessig_fringes(
    Q: np.ndarray,
    R: np.ndarray,
    q_min_analysis: float = 0.02,
    n_fft_points: int = 512,
) -> Dict:
    """
    Extract oscillation periods (Kiessig fringes) using FFT.
    
    The fringe period ΔQ relates to layer thickness:
    d ≈ 2π / ΔQ
    
    Args:
        Q: Q values (Å⁻¹)
        R: Reflectivity values
        q_min_analysis: Minimum Q to use for analysis (skip critical edge)
        n_fft_points: Number of points for FFT
    
    Returns:
        Dict with oscillation information
    """
    # Filter to analysis range (above critical edge)
    mask = Q >= q_min_analysis
    Q_analysis = Q[mask]
    R_analysis = R[mask]
    
    if len(Q_analysis) < 20:
        return {
            'oscillation_periods': [],
            'thicknesses': [],
            'n_fringes': 0,
            'method': 'fft',
        }
    
    # Resample to uniform Q spacing for FFT
    Q_uniform = np.linspace(Q_analysis.min(), Q_analysis.max(), n_fft_points)
    R_uniform = np.interp(Q_uniform, Q_analysis, R_analysis)
    
    # Work in log space (use np.maximum to handle negative values)
    log_R = np.log10(np.maximum(R_uniform, 1e-12))
    
    # Remove trend (Q^-4 decay)
    log_R_detrended = log_R - np.polyval(np.polyfit(Q_uniform, log_R, 1), Q_uniform)
    
    # Apply window to reduce edge effects
    window = np.hanning(len(log_R_detrended))
    log_R_windowed = log_R_detrended * window
    
    # FFT
    fft = np.fft.rfft(log_R_windowed)
    freqs = np.fft.rfftfreq(len(Q_uniform), Q_uniform[1] - Q_uniform[0])
    
    # Power spectrum
    power = np.abs(fft)**2
    
    # Find peaks in power spectrum
    # Skip DC and very low frequencies
    min_freq_idx = 5
    peaks, properties = find_peaks(power[min_freq_idx:], 
                                   prominence=0.1 * np.max(power[min_freq_idx:]))
    peaks = peaks + min_freq_idx
    
    # Convert frequencies to thicknesses
    oscillation_periods = []
    thicknesses = []
    
    for peak_idx in peaks:
        freq = freqs[peak_idx]
        if freq > 0:
            # Period in Q space
            period = 1.0 / freq
            # Thickness: d ≈ 2π / ΔQ
            thickness = 2 * np.pi / period
            
            oscillation_periods.append({
                'frequency': float(freq),
                'period_Q': float(period),
                'amplitude': float(power[peak_idx]),
            })
            thicknesses.append(float(thickness))
    
    # Also count fringes directly
    n_fringes = count_fringes_direct(Q_analysis, R_analysis)
    
    return {
        'oscillation_periods': oscillation_periods,
        'thicknesses': thicknesses,
        'n_fringes': n_fringes,
        'method': 'fft',
    }


def count_fringes_direct(
    Q: np.ndarray,
    R: np.ndarray,
) -> int:
    """
    Count Kiessig fringes by finding local minima in R(Q).
    
    Args:
        Q: Q values
        R: Reflectivity values
    
    Returns:
        Number of fringes detected
    """
    # Use np.maximum to handle negative R values from background subtraction
    log_R = np.log10(np.maximum(R, 1e-12))
    
    # Smooth
    if len(log_R) > 20:
        window = min(11, len(log_R) // 5)
        if window % 2 == 0:
            window += 1
        log_R_smooth = savgol_filter(log_R, window, 2)
    else:
        log_R_smooth = log_R
    
    # Find minima
    minima, _ = find_peaks(-log_R_smooth, distance=5)
    
    return len(minima)


def estimate_total_thickness(
    Q: np.ndarray,
    R: np.ndarray,
    q_min: float = 0.02,
) -> Dict:
    """
    Estimate total film thickness from fringe spacing.
    
    Uses the average spacing between consecutive minima.
    
    Args:
        Q: Q values
        R: Reflectivity values
        q_min: Minimum Q to consider
    
    Returns:
        Dict with thickness estimate and confidence
    """
    mask = Q >= q_min
    Q_analysis = Q[mask]
    R_analysis = R[mask]
    
    if len(Q_analysis) < 20:
        return {
            'thickness': None,
            'uncertainty': None,
            'confidence': 'low',
            'method': 'fringe_spacing',
        }
    
    # Use np.maximum to handle negative R values from background subtraction
    log_R = np.log10(np.maximum(R_analysis, 1e-12))
    
    # Smooth
    window = min(11, len(log_R) // 5)
    if window % 2 == 0:
        window += 1
    log_R_smooth = savgol_filter(log_R, window, 2)
    
    # Find minima
    minima, _ = find_peaks(-log_R_smooth, distance=5)
    
    if len(minima) < 2:
        return {
            'thickness': None,
            'uncertainty': None,
            'confidence': 'low',
            'method': 'fringe_spacing',
            'n_fringes': len(minima),
        }
    
    # Calculate fringe spacings
    Q_minima = Q_analysis[minima]
    delta_Q = np.diff(Q_minima)
    
    # Average fringe spacing
    avg_delta_Q = np.mean(delta_Q)
    std_delta_Q = np.std(delta_Q)
    
    # Thickness from fringe spacing: d ≈ 2π / ΔQ
    thickness = 2 * np.pi / avg_delta_Q
    
    # Uncertainty
    if len(delta_Q) > 1:
        thickness_uncertainty = thickness * (std_delta_Q / avg_delta_Q)
    else:
        thickness_uncertainty = thickness * 0.2  # 20% default uncertainty
    
    # Confidence
    if len(minima) >= 5 and std_delta_Q / avg_delta_Q < 0.1:
        confidence = 'high'
    elif len(minima) >= 3:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    return {
        'thickness': float(thickness),
        'uncertainty': float(thickness_uncertainty),
        'confidence': confidence,
        'method': 'fringe_spacing',
        'n_fringes': len(minima),
        'avg_fringe_spacing': float(avg_delta_Q),
    }


def estimate_roughness(
    Q: np.ndarray,
    R: np.ndarray,
    q_min: float = 0.15,
) -> Dict:
    """
    Estimate interface roughness from high-Q decay.
    
    At high Q, the Debye-Waller factor causes:
    R ∝ R_Fresnel × exp(-Q²σ²)
    
    In log space: log(R) = log(R_F) - Q²σ² / ln(10)
    
    After removing Q⁻⁴ Fresnel decay:
    log(R) + 4*log(Q) = const - Q²σ²/ln(10)
    
    Args:
        Q: Q values
        R: Reflectivity values
        q_min: Minimum Q for roughness estimation
    
    Returns:
        Dict with roughness estimate and confidence
    """
    mask = Q >= q_min
    Q_analysis = Q[mask]
    R_analysis = R[mask]
    
    if len(Q_analysis) < 10:
        return {
            'roughness': 5.0,  # Default guess
            'uncertainty': None,
            'confidence': 'low',
            'method': 'high_q_decay',
        }
    
    # Remove Fresnel decay (Q^-4)
    # Use np.maximum to handle negative R values from background subtraction
    log_R = np.log10(np.maximum(R_analysis, 1e-12))
    log_R_corrected = log_R + 4 * np.log10(Q_analysis)
    
    # Linear fit: log_R_corrected = a - b*Q²
    # where b = σ²/ln(10)
    Q_sq = Q_analysis**2
    
    try:
        coeffs = np.polyfit(Q_sq, log_R_corrected, 1)
        slope = coeffs[0]  # This is -σ²/ln(10)
        
        # Extract sigma
        # Note: slope is negative if there's roughness
        sigma_sq = -slope * np.log(10)
        
        if sigma_sq > 0:
            sigma = np.sqrt(sigma_sq)
            
            # Estimate uncertainty from fit residuals
            residuals = log_R_corrected - np.polyval(coeffs, Q_sq)
            fit_quality = np.std(residuals)
            
            if fit_quality < 0.3:
                confidence = 'high'
            elif fit_quality < 0.6:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            return {
                'roughness': float(sigma),
                'uncertainty': float(sigma * fit_quality),
                'confidence': confidence,
                'method': 'high_q_decay',
                'fit_quality': float(fit_quality),
            }
        else:
            # Negative sigma_sq means no roughness or bad fit
            return {
                'roughness': 0.0,
                'uncertainty': None,
                'confidence': 'low',
                'method': 'high_q_decay',
            }
    
    except Exception:
        return {
            'roughness': 5.0,
            'uncertainty': None,
            'confidence': 'low',
            'method': 'high_q_decay',
        }


def estimate_layer_count(
    Q: np.ndarray,
    R: np.ndarray,
    critical_edges: List[Dict],
    oscillation_info: Dict,
) -> Dict:
    """
    Estimate number of layers from features.
    
    Heuristics:
    - Number of distinct Qc values suggests number of materials
    - Complexity of fringe pattern suggests number of layers
    - Multiple distinct oscillation frequencies suggest multiple layers
    
    Args:
        Q: Q values
        R: Reflectivity values
        critical_edges: Output from extract_critical_edges
        oscillation_info: Output from extract_kiessig_fringes
    
    Returns:
        Dict with layer count estimate and confidence
    """
    # Count indicators
    n_critical_edges = len(critical_edges)
    n_oscillation_freqs = len(oscillation_info.get('thicknesses', []))
    n_fringes = oscillation_info.get('n_fringes', 0)
    
    # Simple heuristic
    # 0 layers (Fresnel): very few/no fringes, 1 critical edge
    # 1 layer: regular fringes, 1-2 critical edges
    # 2 layers: beat pattern in fringes, multiple frequencies
    # 3+ layers: complex pattern
    
    if n_fringes < 2 and n_oscillation_freqs == 0:
        estimated = 0
        confidence = 'medium'
    elif n_oscillation_freqs <= 1 and n_critical_edges <= 2:
        estimated = 1
        confidence = 'medium' if n_fringes >= 3 else 'low'
    elif n_oscillation_freqs == 2 or n_critical_edges == 3:
        estimated = 2
        confidence = 'medium'
    else:
        estimated = min(3, max(n_oscillation_freqs, n_critical_edges - 1))
        confidence = 'low'
    
    return {
        'estimated_n_layers': estimated,
        'confidence': confidence,
        'indicators': {
            'n_critical_edges': n_critical_edges,
            'n_oscillation_freqs': n_oscillation_freqs,
            'n_fringes': n_fringes,
        },
    }


def extract_all_features(
    Q: np.ndarray,
    R: np.ndarray,
    dR: Optional[np.ndarray] = None,
) -> Dict:
    """
    Extract all physics features from reflectivity data.
    
    This is the main entry point for feature extraction.
    
    Args:
        Q: Q values (Å⁻¹)
        R: Reflectivity values
        dR: Optional error values
    
    Returns:
        Comprehensive feature dictionary
    """
    # Critical edges
    critical_edges = extract_critical_edges(Q, R)
    
    # Oscillations/fringes
    oscillation_info = extract_kiessig_fringes(Q, R)
    
    # Total thickness
    thickness_info = estimate_total_thickness(Q, R)
    
    # Roughness
    roughness_info = estimate_roughness(Q, R)
    
    # Layer count
    layer_count = estimate_layer_count(Q, R, critical_edges, oscillation_info)
    
    return {
        # Critical edge information
        'critical_edges': critical_edges,
        'n_critical_edges': len(critical_edges),
        
        # Oscillation information
        'oscillation_periods': oscillation_info.get('oscillation_periods', []),
        'thicknesses': oscillation_info.get('thicknesses', []),
        'n_fringes': oscillation_info.get('n_fringes', 0),
        
        # Total thickness
        'estimated_total_thickness': thickness_info.get('thickness'),
        'thickness_uncertainty': thickness_info.get('uncertainty'),
        'thickness_confidence': thickness_info.get('confidence'),
        
        # Roughness
        'estimated_roughness': roughness_info.get('roughness'),
        'roughness_confidence': roughness_info.get('confidence'),
        
        # Layer count
        'estimated_n_layers': layer_count.get('estimated_n_layers'),
        'layer_count_confidence': layer_count.get('confidence'),
        
        # Data quality
        'q_min': float(Q.min()),
        'q_max': float(Q.max()),
        'n_points': len(Q),
        'has_error_bars': dR is not None,
    }


def format_features_for_llm(features: Dict) -> str:
    """
    Format extracted features as human-readable text for LLM context.
    
    Args:
        features: Output from extract_all_features
    
    Returns:
        Formatted string description
    """
    lines = ["## Extracted Physics Features\n"]
    
    # Data quality
    lines.append("### Data Quality")
    lines.append(f"- Q range: {features['q_min']:.4f} - {features['q_max']:.4f} Å⁻¹")
    lines.append(f"- Number of points: {features['n_points']}")
    lines.append(f"- Has error bars: {features['has_error_bars']}")
    lines.append("")
    
    # Critical edges
    lines.append("### Critical Edge Analysis")
    if features['critical_edges']:
        for i, edge in enumerate(features['critical_edges']):
            lines.append(f"- Edge {i+1}: Qc = {edge['Qc']:.4f} Å⁻¹ → SLD ≈ {edge['estimated_SLD']:.2f} × 10⁻⁶ Å⁻² ({edge['confidence']} confidence)")
    else:
        lines.append("- No clear critical edges detected")
    lines.append("")
    
    # Thickness
    lines.append("### Thickness Analysis")
    if features['estimated_total_thickness']:
        lines.append(f"- Estimated total thickness: {features['estimated_total_thickness']:.1f} ± {features['thickness_uncertainty']:.1f} Å ({features['thickness_confidence']} confidence)")
    else:
        lines.append("- Could not estimate thickness from fringe pattern")
    lines.append(f"- Number of fringes detected: {features['n_fringes']}")
    
    if features['thicknesses']:
        lines.append("- Individual layer thickness candidates:")
        for d in features['thicknesses'][:5]:  # Limit to 5
            lines.append(f"  - {d:.1f} Å")
    lines.append("")
    
    # Roughness
    lines.append("### Roughness Analysis")
    if features['estimated_roughness']:
        lines.append(f"- Estimated roughness: {features['estimated_roughness']:.1f} Å ({features['roughness_confidence']} confidence)")
    else:
        lines.append("- Could not estimate roughness from high-Q decay")
    lines.append("")
    
    # Layer count
    lines.append("### Layer Count Estimate")
    lines.append(f"- Estimated number of layers: {features['estimated_n_layers']} ({features['layer_count_confidence']} confidence)")
    
    return "\n".join(lines)


if __name__ == '__main__':
    # Test with synthetic data
    print("Feature extraction tools ready.")
    print("Run with: python -m aure.tools.feature_tools")
