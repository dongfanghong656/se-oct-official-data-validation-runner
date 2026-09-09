# Scientific claim matrix

This matrix separates software execution from physical interpretation. A narrower image is not, by itself, proof of physical out-of-band recovery.

| Claim | Current status | Strongest current evidence | Missing promotion gate |
|---|---|---|---|
| Deposited FIAA/MIAA implements complex covariance-based missing-spectrum prediction | **VERIFIED_CONDITIONAL** | Equation implementation and selected high-SNR public A-lines | Broader independent implementation comparison |
| MIAA is merely final-image unsharp masking | **REJECTED_AS_COMPLETE_EXPLANATION** | Complex heldout prediction and phase-sensitive controls | None for this narrow claim |
| MIAA automatically estimates and removes dispersion | **REJECTED_FOR_TESTED_PHASE_FAMILIES** | Injected quadratic/cubic phase requires explicit inverse compensation | Raw-camera and spatially varying sample-dispersion audit |
| Central high-SNR bins predict nearby bins of the same deposited finite-depth representation | **SUPPORTED_CONDITIONALLY** | Internal complex holdout | Independent physical wideband spectrum |
| All far extrapolated bins equal a real wider-band OCT measurement | **NOT_ESTABLISHED** | No independent outer-band acquisition | Frozen-parameter 100-nm/50-nm blind complex validation |
| A 1–2 micrometre single-point core is a universal linear-system axial resolution | **NOT_ESTABLISHED** | Single-particle Gaussian FWHM | Registered two-target, phase/amplitude/SNR and false-peak study |
| MIAA output has one object-independent convolution PSF | **NOT_ESTABLISHED / GENERALLY NONLINEAR** | Data-adaptive covariance implies object dependence | Local additivity and dense-object validation |
| RFIAA warm-start recursion is scan-order invariant | **UNDER_AUDIT** | Forward/reverse/independent compact-subset audit | Full B-scan and repeated-scan confirmation |
| `super=4` is uniquely selected by measured data | **NOT_ESTABLISHED** | Internal support-factor audit only | Independent wideband likelihood or registered uncertainty rule |
| Extrapolated phase is sufficiently coherent for some ISAM/CAO processing | **SUPPORTED_ON_SELECTED_DATA** | Experimental refocusing and phase-scramble controls | Full 3-D paired phase-error and energy audit |
| Public data can reproduce raw k-linearization, dispersion and coverslip correction | **BLOCKED_BY_UPSTREAM_DATA_ABSENCE** | Deposited inputs are already phase corrected | Raw camera interferograms and calibration frames |
| Figure 5 proves artifact-free dense-tissue super-resolution | **NOT_ESTABLISHED** | Selected A-line and visual sectioning improvement | Dense-object truth, topology/false-structure metrics, common display scale |
| CAO improvement proves MIAA out-of-band phase is physically correct | **REJECTED_AS_LOGICAL_INFERENCE** | CAO is itself an adaptive wavefront optimization | Independent wavefront or held-out spatial validation |

Release remains **NOT_RELEASED** until the physical-wideband, two-target, full-volume phase and raw-preprocessing gates are satisfied.
