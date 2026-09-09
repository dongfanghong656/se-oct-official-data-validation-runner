## TURN-0001 upstream-data validation result

Workflow run: https://github.com/dongfanghong656/se-oct-official-data-validation-runner/actions/runs/34237243155

**Data boundary:** Zenodo's earliest public experimental input is already phase corrected; it is not the raw camera interferogram. This run therefore validates post-correction spectral-estimation behavior, not the authors' original k-linearization/dispersion/coverslip correction.

### Official TiO₂ selected-A-line controls (median)

| Method / metric | Result |
|---|---:|
| DFT from central 128 bins, FWHM | 5.716 µm |
| DFT from all deposited 200 bins, FWHM | 4.080 µm |
| IAA from central 128 bins, FWHM | 1.266 µm |
| MIAA 128→200, FWHM | 3.713 µm |
| Complex AR 128→200, FWHM | 3.756 µm |
| MIAA held-out complex correlation | 0.9744 |
| Complex AR held-out correlation | 0.9761 |
| MIAA held-out phase RMSE | 0.0909 rad |
| Injected-dispersion MIAA similarity to baseline | 0.2565 |
| Explicitly corrected MIAA similarity to baseline | 1.0000 |

### Known-ground-truth MIAA→ISAM coherence control

| Method | Lateral FWHM | Relative peak amplitude | Local energy fraction |
|---|---:|---:|---:|
| `conventional_narrow_no_isam` | 7.273 µm | 0.2879 | 0.5838 |
| `full_measured_isam` | 1.492 µm | 1.0000 | 0.9998 |
| `narrow_zero_fill_isam` | 1.490 µm | 0.3863 | 0.9998 |
| `miaa_then_isam` | 1.496 µm | 0.9535 | 0.9997 |
| `phase_scrambled_miaa_then_isam` | 2.254 µm | 0.3763 | 0.2324 |

### Interpretation gate

- MIAA is **not accepted as dispersion compensation** merely because a peak narrows. The injected-quadratic-phase control must remain degraded until an explicit inverse spectral phase is applied.
- MIAA is **not accepted as generic image enhancement** merely because FWHM shrinks. It must predict physically held-out complex spectral samples and retain the cross-A-line phase needed by ISAM.
- Any extrapolation beyond the deposited 200-bin spectrum remains model-dependent until compared with a genuinely wider measured spectrum.

Full JSON, figures, test logs, and upstream-code audit are attached to the workflow artifact `seoct-validation-evidence`.