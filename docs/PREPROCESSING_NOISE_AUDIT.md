# MIAA preprocessing and demodulated-noise audit

## Scope

This audit isolates two preprocessing questions in the deposited pipeline:

1. Does the order of the complex scalar product in the source-shaped background subtraction explain the MIAA sharpening?
2. What noise covariance reaches FIAA/MIAA after division by `sk`, selection of the 128-bin source window, and per-A-line mean subtraction?

The inputs are the committed, checksum-traced compact TiO2 subset and exact selected-A-line artifact. They derive from the authors' already phase-corrected C-scan, not raw camera interferograms.

## Algebraic result

The deposited code forms `C = raw - g*sk`, then `anC = C./sk`, selects the input band, and subtracts the per-A-line mean. For any scalar `g` and nonzero matching source samples,

```text
Q[(raw - g*sk)./sk] = Q[raw./sk]
```

because `Q*1 = 0`. Thus the deposited `raw' * sk` coefficient and the standard complex least-squares `sk' * raw` coefficient produce the same MIAA input to numerical precision. The coefficient difference cannot be the hidden cause of the MIAA narrow core. The DFT branch is not algebraically invariant because it uses `detrend(C)` before source division, so it is audited separately.

## Exact noise operator

Let `P` select the source window, `D = diag(1/sk)`, and `Q = I - 11^T/N` subtract the uniform mean. The MIAA input noise covariance is

```text
Sigma_demod = Q D P Sigma_raw P^H D^H Q^H.
```

Even when `Sigma_raw` is white, source division makes the variance heteroscedastic and `Q` introduces correlations and one null mode. A scalar `eta*I` is therefore a computational approximation, not the exact covariance of the deposited preprocessing chain.

## Interpretation boundary

The official-line comparison reuses IAA power and eta fitted under the author's scalar-noise implementation. Substituting a trace-matched full covariance without refitting power is a sensitivity audit, not a corrected reconstruction. The oracle SNR sweep supplies true power and true covariance only to establish the possible low-SNR consequence of the approximation.
