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
from typing import Dict, List, Optional


def extract_critical_edges(
    Q: np.ndarray,
    R: np.ndarray,
    min_qc: float = 0.005,
    max_qc: float = 0.05,
) -> List[Dict]:
    """
    Find the critical edge and estimate the corresponding SLD.

    The critical edge Qc is where total external reflection ends and
    reflectivity drops from the initial plateau at low Q.

    Qc is estimated at the **half-height of the total-reflection plateau**:
    under resolution smearing R falls to ~½ of the plateau level at the true
    Qc, so this is both accurate and robust to a plateau that sits below 1.0
    (imperfect intensity normalization).  The search excludes any region where
    Kiessig fringes (oscillations) have already begun.  At most one candidate
    is returned.

    For total external reflection: Qc = 4·sqrt(π·SLD)

    Args:
        Q: Q values (Å⁻¹)
        R: Reflectivity values
        min_qc: Minimum Qc to consider (Å⁻¹)
        max_qc: Maximum Qc to consider (Å⁻¹)

    Returns:
        List of dicts with {Qc, estimated_SLD, confidence, gradient}
    """
    log_R = np.log10(np.maximum(R, 1e-12))

    # Smooth to reduce noise
    if len(log_R) > 20:
        window = min(11, len(log_R) // 5)
        if window % 2 == 0:
            window += 1
        log_R_smooth = savgol_filter(log_R, window, 3)
    else:
        log_R_smooth = log_R.copy()

    dlogR_dQ = np.gradient(log_R_smooth, Q)

    # --- Upper Q boundary: first fringe trough --------------------------
    # A fringe trough is a local minimum in log_R_smooth that is followed
    # by a recovery (what makes find_peaks detect it).  Prominence ≥ 0.1
    # in log10(R) filters out noise while catching real fringes.
    # Anything at or after the first fringe cannot be a critical edge.
    fringe_minima, _ = find_peaks(-log_R_smooth, distance=5, prominence=0.1)

    upper_q = max_qc
    for mi in fringe_minima:
        if Q[mi] >= min_qc:
            upper_q = min(upper_q, Q[mi])
            break

    # --- Locate the critical edge in [min_qc, upper_q] ------------------
    search_mask = (Q >= min_qc) & (Q <= upper_q)
    if not np.any(search_mask):
        # Fallback: use the full Qc range
        search_mask = (Q >= min_qc) & (Q <= max_qc)
    if not np.any(search_mask):
        return []

    search_idx = np.where(search_mask)[0]
    grad_in_region = dlogR_dQ[search_idx]

    # If nothing drops there is no discernible edge.
    min_grad = float(np.min(grad_in_region))
    if min_grad >= -1.0:
        return []

    R_smooth = 10.0**log_R_smooth

    # Plateau level: robust (median) reflectivity over a small low-Q window at
    # the start of the search region — the total-reflection plateau. Estimated
    # from the low-Q points directly (NOT from the steepest-gradient point,
    # which for a curve decaying toward zero sits far above the edge, deep in
    # the tail). Using the measured plateau — not an assumed R=1 — makes Qc
    # correct even when the plateau sits below 1.0 (imperfect normalization).
    n_plateau = int(min(max(3, len(search_idx) // 8), 12))
    plateau = float(np.median(R_smooth[search_idx[:n_plateau]]))
    half_level = 0.5 * plateau

    # Qc = first crossing below half the plateau, scanning up from low Q. This
    # is the true critical edge; the previous "onset of the first drop"
    # heuristic fired on plateau noise and under-reported Qc.
    cross_idx = None
    for k in search_idx:
        if R_smooth[k] < half_level:
            cross_idx = int(k)
            break

    if cross_idx is None or cross_idx == int(search_idx[0]):
        # No plateau captured (data may start above Qc) — fall back to the
        # steepest-descent point in the region.
        steep_idx = int(search_idx[int(np.argmin(grad_in_region))])
        qc = float(Q[steep_idx])
        qc_idx = steep_idx
    else:
        # Linear-interpolate the crossing between the bracketing points.
        prev = cross_idx - 1
        if prev >= 0 and R_smooth[prev] != R_smooth[cross_idx]:
            frac = (R_smooth[prev] - half_level) / (
                R_smooth[prev] - R_smooth[cross_idx]
            )
            qc = float(Q[prev] + frac * (Q[cross_idx] - Q[prev]))
        else:
            qc = float(Q[cross_idx])
        qc_idx = cross_idx

    # SLD contrast from Qc:  Δρ = (Qc / 4)² / π  × 10⁶  (incident medium = 0)
    sld = (qc / 4) ** 2 / np.pi * 1e6

    edge_sharpness = abs(dlogR_dQ[qc_idx])
    if edge_sharpness > 50:
        confidence = "high"
    elif edge_sharpness > 20:
        confidence = "medium"
    else:
        confidence = "low"

    return [
        {
            "Qc": float(qc),
            "estimated_SLD": float(sld),
            "confidence": confidence,
            "gradient": float(dlogR_dQ[qc_idx]),
        }
    ]


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
            "oscillation_periods": [],
            "n_fringes": 0,
            "method": "fft",
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
    power = np.abs(fft) ** 2

    # Find peaks in power spectrum
    # Skip DC and very low frequencies
    min_freq_idx = 5
    peaks, properties = find_peaks(
        power[min_freq_idx:], prominence=0.1 * np.max(power[min_freq_idx:])
    )
    peaks = peaks + min_freq_idx

    # Convert frequencies to oscillation periods
    oscillation_periods = []

    for peak_idx in peaks:
        freq = freqs[peak_idx]
        if freq > 0:
            # Period in Q space
            period = 1.0 / freq

            oscillation_periods.append(
                {
                    "frequency": float(freq),
                    "period_Q": float(period),
                    "amplitude": float(power[peak_idx]),
                }
            )

    # Also count fringes directly
    n_fringes = count_fringes_direct(Q_analysis, R_analysis)

    return {
        "oscillation_periods": oscillation_periods,
        "n_fringes": n_fringes,
        "method": "fft",
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

    # Find minima — require prominence to reject noise wiggles
    minima, _ = find_peaks(-log_R_smooth, distance=5, prominence=0.05)

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
            "thickness": None,
            "uncertainty": None,
            "confidence": "low",
            "method": "fringe_spacing",
        }

    # Use np.maximum to handle negative R values from background subtraction
    log_R = np.log10(np.maximum(R_analysis, 1e-12))

    # Smooth
    window = min(11, len(log_R) // 5)
    if window % 2 == 0:
        window += 1
    log_R_smooth = savgol_filter(log_R, window, 2)

    # Find minima — require prominence to reject noise wiggles
    minima, _ = find_peaks(-log_R_smooth, distance=5, prominence=0.05)

    if len(minima) < 2:
        return {
            "thickness": None,
            "uncertainty": None,
            "confidence": "low",
            "method": "fringe_spacing",
            "n_fringes": len(minima),
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
        confidence = "high"
    elif len(minima) >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "thickness": float(thickness),
        "uncertainty": float(thickness_uncertainty),
        "confidence": confidence,
        "method": "fringe_spacing",
        "n_fringes": len(minima),
        "avg_fringe_spacing": float(avg_delta_Q),
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
            "roughness": 5.0,  # Default guess
            "uncertainty": None,
            "confidence": "low",
            "method": "high_q_decay",
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
                confidence = "high"
            elif fit_quality < 0.6:
                confidence = "medium"
            else:
                confidence = "low"

            return {
                "roughness": float(sigma),
                "uncertainty": float(sigma * fit_quality),
                "confidence": confidence,
                "method": "high_q_decay",
                "fit_quality": float(fit_quality),
            }
        else:
            # Negative sigma_sq means no roughness or bad fit
            return {
                "roughness": 0.0,
                "uncertainty": None,
                "confidence": "low",
                "method": "high_q_decay",
            }

    except Exception:
        return {
            "roughness": 5.0,
            "uncertainty": None,
            "confidence": "low",
            "method": "high_q_decay",
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
    n_oscillation_freqs = len(oscillation_info.get("oscillation_periods", []))
    n_fringes = oscillation_info.get("n_fringes", 0)

    # Simple heuristic
    # 0 layers (Fresnel): very few/no fringes, 1 critical edge
    # 1 layer: regular fringes, 1-2 critical edges
    # 2 layers: beat pattern in fringes, multiple frequencies
    # 3+ layers: complex pattern

    if n_fringes < 2 and n_oscillation_freqs == 0:
        estimated = 0
        confidence = "medium"
    elif n_oscillation_freqs <= 1 and n_critical_edges <= 2:
        estimated = 1
        confidence = "medium" if n_fringes >= 3 else "low"
    elif n_oscillation_freqs == 2 or n_critical_edges == 3:
        estimated = 2
        confidence = "medium"
    else:
        estimated = min(3, max(n_oscillation_freqs, n_critical_edges - 1))
        confidence = "low"

    return {
        "estimated_n_layers": estimated,
        "confidence": confidence,
        "indicators": {
            "n_critical_edges": n_critical_edges,
            "n_oscillation_freqs": n_oscillation_freqs,
            "n_fringes": n_fringes,
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
    # De-duplicate Q values (merged datasets can have repeats, which
    # cause np.gradient to produce NaN via division by zero).
    _, unique_idx = np.unique(Q, return_index=True)
    if len(unique_idx) < len(Q):
        Q = Q[unique_idx]
        R = R[unique_idx]
        if dR is not None:
            dR = dR[unique_idx]

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
        "critical_edges": critical_edges,
        "n_critical_edges": len(critical_edges),
        # Oscillation information
        "oscillation_periods": oscillation_info.get("oscillation_periods", []),
        "n_fringes": oscillation_info.get("n_fringes", 0),
        # Total thickness
        "estimated_total_thickness": thickness_info.get("thickness"),
        "thickness_uncertainty": thickness_info.get("uncertainty"),
        "thickness_confidence": thickness_info.get("confidence"),
        # Roughness
        "estimated_roughness": roughness_info.get("roughness"),
        "roughness_confidence": roughness_info.get("confidence"),
        # Layer count
        "estimated_n_layers": layer_count.get("estimated_n_layers"),
        "layer_count_confidence": layer_count.get("confidence"),
        # Data quality
        "q_min": float(Q.min()),
        "q_max": float(Q.max()),
        "n_points": len(Q),
        "has_error_bars": dR is not None,
    }


def analyze_residual_fringes(
    Q: np.ndarray,
    residual_ratio: np.ndarray,
    q_min: float = 0.02,
    q_max: float | None = None,
    n_fft_points: int = 1024,
) -> Dict:
    """
    Detect unmodeled layer thicknesses from oscillations in the residual ratio.

    When R_data / R_fit oscillates around 1.0 with periodic structure, it
    indicates the model is missing a layer whose thickness determines the
    oscillation period via d ≈ 2π / ΔQ.

    Args:
        Q: Q values (Å⁻¹), same grid as residual_ratio
        residual_ratio: R_data / R_fit array
        q_min: Minimum Q to analyze (skip critical edge region)
        q_max: Maximum Q (skip noisy high-Q tail); None = use all
        n_fft_points: Number of FFT points

    Returns:
        Dict with:
          - has_residual_fringes: bool
          - unmodeled_thicknesses: list of {thickness, uncertainty, confidence, method}
          - fringe_amplitude: float (RMS deviation from 1.0)
          - n_residual_fringes: int
    """
    Q = np.asarray(Q, dtype=float)
    residual_ratio = np.asarray(residual_ratio, dtype=float)

    if len(Q) != len(residual_ratio) or len(Q) < 20:
        return _empty_residual_result()

    # Apply Q range filter
    mask = Q >= q_min
    if q_max is not None:
        mask &= Q <= q_max
    Q_sel = Q[mask]
    ratio_sel = residual_ratio[mask]

    if len(Q_sel) < 20:
        return _empty_residual_result()

    # Fringe amplitude: RMS deviation of ratio from 1.0
    fringe_amplitude = float(np.sqrt(np.mean((ratio_sel - 1.0) ** 2)))

    # If the residual is very flat, no fringes to find
    if fringe_amplitude < 0.02:
        return {
            "has_residual_fringes": False,
            "unmodeled_thicknesses": [],
            "fringe_amplitude": fringe_amplitude,
            "n_residual_fringes": 0,
        }

    thicknesses = []

    # --- Method 1: FFT on the ratio signal ---
    fft_thicknesses = _fft_residual_thicknesses(Q_sel, ratio_sel, n_fft_points)
    thicknesses.extend(fft_thicknesses)

    # --- Method 2: Direct fringe-spacing on the ratio signal ---
    spacing_result = _fringe_spacing_residual(Q_sel, ratio_sel)
    if spacing_result is not None:
        thicknesses.append(spacing_result)

    # Deduplicate: merge results within 20% of each other
    thicknesses = _deduplicate_thicknesses(thicknesses)

    n_fringes = max((t.get("n_fringes", 0) for t in thicknesses), default=0)

    return {
        "has_residual_fringes": len(thicknesses) > 0,
        "unmodeled_thicknesses": thicknesses,
        "fringe_amplitude": fringe_amplitude,
        "n_residual_fringes": n_fringes,
    }


def _empty_residual_result() -> Dict:
    """Return an empty residual analysis result."""
    return {
        "has_residual_fringes": False,
        "unmodeled_thicknesses": [],
        "fringe_amplitude": 0.0,
        "n_residual_fringes": 0,
    }


def _fft_residual_thicknesses(
    Q: np.ndarray,
    ratio: np.ndarray,
    n_fft_points: int = 1024,
) -> List[Dict]:
    """Extract unmodeled thicknesses via FFT of the residual ratio signal."""
    # Resample to uniform Q spacing
    Q_uniform = np.linspace(Q.min(), Q.max(), n_fft_points)
    ratio_uniform = np.interp(Q_uniform, Q, ratio)

    # Polynomial detrending (order 3) to remove low-frequency model-mismatch
    # envelope while preserving oscillatory fringe signal
    coeffs = np.polyfit(Q_uniform, ratio_uniform, 3)
    deviation = ratio_uniform - np.polyval(coeffs, Q_uniform)

    # Apply Hanning window
    window = np.hanning(len(deviation))
    deviation_windowed = deviation * window

    # FFT
    fft_vals = np.fft.rfft(deviation_windowed)
    freqs = np.fft.rfftfreq(len(Q_uniform), Q_uniform[1] - Q_uniform[0])
    power = np.abs(fft_vals) ** 2

    # Skip frequencies below the minimum detectable: require at least
    # min_fringes complete oscillations visible in the Q range
    Q_range = Q.max() - Q.min()
    min_fringes = 5
    min_thickness = min_fringes * 2.0 * np.pi / Q_range
    freq_step = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    min_freq_idx = max(3, int((min_thickness / (2.0 * np.pi)) / (freq_step + 1e-30)))
    if min_freq_idx >= len(power):
        return []

    power_search = power[min_freq_idx:]
    if len(power_search) < 3 or np.max(power_search) == 0:
        return []

    peaks, properties = find_peaks(
        power_search,
        prominence=0.05 * np.max(power_search),
        distance=2,
    )

    results = []
    for peak in peaks:
        idx = peak + min_freq_idx
        freq = freqs[idx]
        if freq <= 0:
            continue
        # freq from rfftfreq is in cycles per Å⁻¹ (units of Å)
        # Kiessig: thickness = 2π / ΔQ, and ΔQ = 1/freq → thickness = 2π * freq
        thickness_angstrom = 2.0 * np.pi * freq

        if thickness_angstrom < 50:  # unphysically thin
            continue

        results.append(
            {
                "thickness": float(thickness_angstrom),
                "uncertainty": float(thickness_angstrom * 0.15),
                "confidence": "medium",
                "method": "residual_fft",
                "fft_power": float(power[idx]),
            }
        )

    # Sort by FFT power (most significant first), keep top 5
    results.sort(key=lambda x: x.get("fft_power", 0), reverse=True)
    return results[:5]


def _fringe_spacing_residual(Q: np.ndarray, ratio: np.ndarray) -> Optional[Dict]:
    """Extract dominant unmodeled thickness via fringe-spacing on the ratio."""
    if len(ratio) < 20:
        return None

    # Polynomial detrending to remove low-frequency model-mismatch envelope
    coeffs = np.polyfit(Q, ratio, 3)
    deviation = ratio - np.polyval(coeffs, Q)

    # Use a small smoothing window to preserve closely-spaced fringes
    # (thick layers produce fringes spaced only a few data points apart)
    window = min(5, len(deviation) // 10)
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    smooth = savgol_filter(deviation, window, 2)

    # Find minima of the smoothed deviation (troughs in ratio)
    # Use a low prominence threshold relative to the MAD (median absolute
    # deviation) to detect small fringes masked by residual envelope
    mad = np.median(np.abs(smooth - np.median(smooth)))
    if mad < 0.005:
        return None
    prominence = max(0.5 * mad, 0.01)

    minima, _ = find_peaks(-smooth, distance=3, prominence=prominence)

    if len(minima) < 3:
        return None

    delta_Q = np.diff(Q[minima])
    # Use median spacing (robust to outlier spacings from false minima)
    median_delta_Q = float(np.median(delta_Q))
    if median_delta_Q <= 0:
        return None

    thickness = 2.0 * np.pi / median_delta_Q
    n_fringes = len(minima)

    # Uncertainty from spread in fringe spacings
    if len(delta_Q) > 1:
        # Use MAD of spacings for robust uncertainty
        spacing_mad = np.median(np.abs(delta_Q - median_delta_Q))
        uncertainty = (
            thickness * (spacing_mad / median_delta_Q)
            if median_delta_Q > 0
            else thickness * 0.2
        )
    else:
        uncertainty = thickness * 0.2

    # Confidence
    if n_fringes >= 5 and len(delta_Q) > 1 and np.std(delta_Q) / median_delta_Q < 0.3:
        confidence = "high"
    elif n_fringes >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "thickness": float(thickness),
        "uncertainty": float(uncertainty),
        "confidence": confidence,
        "method": "residual_fringe_spacing",
        "n_fringes": n_fringes,
    }


def _deduplicate_thicknesses(thicknesses: List[Dict]) -> List[Dict]:
    """Merge thickness estimates within 20% of each other, keeping the highest confidence."""
    if not thicknesses:
        return []

    confidence_rank = {"high": 3, "medium": 2, "low": 1}
    # Sort by confidence (best first), then by FFT power / n_fringes (highest first)
    thicknesses.sort(
        key=lambda t: (
            -confidence_rank.get(t.get("confidence", "low"), 0),
            -t.get("fft_power", 0),
            -t.get("n_fringes", 0),
        )
    )

    merged = []
    for t in thicknesses:
        found_match = False
        for m in merged:
            if abs(t["thickness"] - m["thickness"]) / max(m["thickness"], 1) < 0.2:
                # Keep the higher-confidence estimate
                if confidence_rank.get(t.get("confidence"), 0) > confidence_rank.get(
                    m.get("confidence"), 0
                ):
                    m.update(t)
                found_match = True
                break
        if not found_match:
            merged.append(dict(t))

    # Remove internal fft_power key from results
    for m in merged:
        m.pop("fft_power", None)

    return merged


def format_critical_edge_line(edge: Dict) -> str:
    """One-line description of a critical edge, incl. any ambient-SLD hint.

    Shared by the analysis display and the evaluation/modeling prompts so the
    deterministic "critical edge implies a deuterated ambient" hint (annotated
    onto the edge by the analysis node) travels wherever critical edges are
    shown.
    """
    line = (
        f"Qc = {edge.get('Qc', 0):.4f} Å⁻¹ → SLD contrast ≈ "
        f"{edge.get('estimated_SLD', 0):.2f} × 10⁻⁶ Å⁻² "
        f"({edge.get('confidence', '?')} confidence)"
    )
    implied = edge.get("implied_ambient_sld")
    if edge.get("suggests_deuteration") and implied is not None:
        sub = edge.get("substrate_sld")
        sub_str = f"{sub:.2f}" if isinstance(sub, (int, float)) else "?"
        line += (
            f" — with the substrate (SLD {sub_str}) this implies an ambient SLD "
            f"≈ {implied:.1f}, far above the H-form (~0): the ambient is very "
            f"likely DEUTERATED. Constrain the ambient SLD near {implied:.1f}, "
            f"not across the full H–D range."
        )
    return line


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
    if features["critical_edges"]:
        for i, edge in enumerate(features["critical_edges"]):
            lines.append(f"- Edge {i + 1}: {format_critical_edge_line(edge)}")
    else:
        lines.append("- No clear critical edges detected")
    lines.append("")

    # Thickness
    lines.append("### Thickness Analysis")
    if features["estimated_total_thickness"]:
        lines.append(
            f"- Estimated total thickness: {features['estimated_total_thickness']:.1f} ± {features['thickness_uncertainty']:.1f} Å ({features['thickness_confidence']} confidence)"
        )
    else:
        lines.append("- Could not estimate thickness from fringe pattern")
    lines.append(f"- Number of fringes detected: {features['n_fringes']}")
    lines.append("")

    # Roughness
    lines.append("### Roughness Analysis")
    if features["estimated_roughness"]:
        lines.append(
            f"- Estimated roughness: {features['estimated_roughness']:.1f} Å ({features['roughness_confidence']} confidence)"
        )
    else:
        lines.append("- Could not estimate roughness from high-Q decay")
    lines.append("")

    # Layer count
    lines.append("### Layer Count Estimate")
    lines.append(
        f"- Estimated number of layers: {features['estimated_n_layers']} ({features['layer_count_confidence']} confidence)"
    )

    return "\n".join(lines)


if __name__ == "__main__":
    # Test with synthetic data
    print("Feature extraction tools ready.")
    print("Run with: python -m aure.tools.feature_tools")
