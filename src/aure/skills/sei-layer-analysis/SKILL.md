---
name: sei-layer-analysis
description: >
  Domain knowledge for solid electrolyte interphase (SEI) and battery electrode
  interface analysis. Covers SEI layer thickness ranges, lithium plating, battery
  material SLDs, and electrochemical cell conventions. Use when the sample involves
  batteries, electrochemistry, SEI, lithium plating, anodes, or cathodes.
metadata:
  author: aure
  version: "1.0"
---

## SEI Layer Properties

The solid electrolyte interphase (SEI) forms between the electrolyte and the
electrode surface in battery systems.

- **Typical thickness**: 50–200 Å
- **SLD range**: 1.0–4.0 × 10⁻⁶ Å⁻² (composition-dependent)
- **Thickness bounds for fitting**: 5–500 Å
- **Position**: between the electrolyte (ambient) and the electrode surface

When adding an SEI layer:
- Set initial thickness to 100 Å (midpoint of typical range)
- Use wide SLD bounds (0.5–5.0) since SEI composition varies widely
- NEVER add with thickness < 5 Å — such layers cannot be resolved

## Lithium Plating

Lithium metal plating appears between the SEI and the electrode (e.g., copper).

- **Typical thickness**: 10–100 Å
- **SLD**: −0.9 × 10⁻⁶ Å⁻² (pure Li)
- **SLD range for fitting**: −1.5 to 0.5 × 10⁻⁶ Å⁻²
- **Thickness bounds**: 5–300 Å

## Battery Material SLDs (×10⁻⁶ Å⁻²)

| Material | SLD |
|----------|-----|
| Lithium metal | -0.9 |
| LiF | 2.25 |
| Li₂CO₃ | 2.0 |
| Li₂O | 1.4 |
| LiPF₆ (in solution) | ~2.5 |

## Electrochemical Cell Conventions

- If the description says the native oxide "reduces away", do NOT include an
  oxide layer; replace it with the hypothesized SEI/plating layer(s).
- Layer ordering for a typical battery electrode (substrate to ambient):
  substrate → electrode metal → (plating) → SEI → electrolyte (ambient)
- Back-reflection geometry is common for in-situ electrochemical cells
  (neutrons through the substrate).

## Electrolyte Solvents

Common battery electrolyte solvents and their SLDs:

| Solvent | SLD (H) | SLD (D) |
|---------|---------|---------|
| EC (ethylene carbonate) | ~1.0 | ~5.5 |
| DMC (dimethyl carbonate) | ~0.5 | ~5.0 |
| THF | 0.18 | 6.35 (d8-THF) |
