#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from seoct_uncertainty import conditional_miaa, normalized_noise_covariance

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "official_uncertainty_audit"
OUT.mkdir(parents=True, exist_ok=True)


def fourier(indices: np.ndarray, object_grid: int) -> np.ndarray:
    z = np.arange(object_grid)
    return np.exp(-1j * 2 * np.pi * indices[:, None] * z[None, :] / object_grid)


def as_2d(value: np.ndarray, rows: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1:
        array = array[:, None]
    if rows is not None and array.shape[0] != rows and array.shape[1] == rows:
        array = array.T
    return array


def main() -> None:
    source_path = ROOT / "reports" / "exact" / "author_exact_alines.mat"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    data = loadmat(source_path, squeeze_me=True)
    K = int(np.asarray(data["K"]).item())
    demod = as_2d(np.asarray(data["demod_spectrum"], np.complex128))
    reflectivity = as_2d(np.asarray(data["rfiaa_profile"], np.complex128), K)
    pe = as_2d(np.asarray(data["pe_first"], np.float64))
    source = np.asarray(data["sk"], np.float64).reshape(-1)
    ii = np.asarray(data["ii"], dtype=int).reshape(-1) - 1
    if demod.shape[0] != ii.size:
        if demod.shape[1] == ii.size:
            demod = demod.T
        else:
            raise ValueError("demod_spectrum is inconsistent with ii")
    if pe.shape[1] != demod.shape[1] and pe.shape[0] == demod.shape[1]:
        pe = pe.T

    Ng = demod.shape[0]
    given_indices = np.arange(Ng)
    full_target = np.arange(-336, 464)
    missing_all = full_target[(full_target < 0) | (full_target >= Ng)]
    take = np.unique(np.rint(np.linspace(0, missing_all.size - 1, 240)).astype(int))
    missing_indices = missing_all[take]
    distance = np.where(missing_indices < 0, -missing_indices, missing_indices - (Ng - 1)).astype(float)
    fg = fourier(given_indices, K)
    fm = fourier(missing_indices, K)
    source_given = np.abs(source[ii])
    source_given /= max(float(np.max(source_given)), 1e-30)

    line_results = []
    scalar_predictability = []
    hetero_predictability = []
    scalar_snr = []
    hetero_snr = []
    for line in range(demod.shape[1]):
        power = np.abs(reflectivity[:, line]) ** 2
        eta = float(max(pe[-1, line], 1e-14))
        sigma_scalar = np.eye(Ng) * eta
        raw_variance = eta * float(np.median(source_given**2))
        sigma_hetero = normalized_noise_covariance(source_given, raw_noise_variance=raw_variance, relative_floor=0.03)
        scalar = conditional_miaa(demod[:, line], fg, fm, power, sigma_scalar, max_condition_number=1e12)
        hetero = conditional_miaa(demod[:, line], fg, fm, power, sigma_hetero, max_condition_number=1e12)
        scalar_predictability.append(scalar.predictability)
        hetero_predictability.append(hetero.predictability)
        scalar_snr.append(scalar.predicted_snr)
        hetero_snr.append(hetero.predicted_snr)

        def extent(result, q=0.5, snr=1.0):
            mask = (result.predictability >= q) & (result.predicted_snr >= snr)
            return float(np.max(distance[mask])) if np.any(mask) else 0.0

        line_results.append({
            "line": line,
            "eta_scalar": eta,
            "scalar": {
                "median_predictability": float(np.median(scalar.predictability)),
                "minimum_predictability": float(np.min(scalar.predictability)),
                "maximum_confident_distance_bins": extent(scalar),
                "condition_number": scalar.condition_number,
                "jitter_used": scalar.jitter_used,
            },
            "heteroscedastic": {
                "median_predictability": float(np.median(hetero.predictability)),
                "minimum_predictability": float(np.min(hetero.predictability)),
                "maximum_confident_distance_bins": extent(hetero),
                "condition_number": hetero.condition_number,
                "jitter_used": hetero.jitter_used,
            },
            "model_difference": {
                "predictability_median_absolute_difference": float(np.median(np.abs(scalar.predictability-hetero.predictability))),
                "mean_prediction_complex_correlation": float(abs(np.vdot(scalar.mean,hetero.mean))/max(np.linalg.norm(scalar.mean)*np.linalg.norm(hetero.mean),1e-30)),
            },
        })

    scalar_predictability = np.asarray(scalar_predictability)
    hetero_predictability = np.asarray(hetero_predictability)
    scalar_snr = np.asarray(scalar_snr)
    hetero_snr = np.asarray(hetero_snr)
    report = {
        "schema": "official-phase-corrected-miaa-uncertainty-audit-v1",
        "status": "VERIFIED_CONDITIONAL_MODEL_AUDIT",
        "source": {"path": source_path.as_posix(), "n_lines": int(demod.shape[1]), "given_bins": int(Ng), "object_grid": K, "sampled_missing_bins": int(missing_indices.size)},
        "aggregate": {
            "scalar_noise_median_predictability": float(np.median(scalar_predictability)),
            "heteroscedastic_noise_median_predictability": float(np.median(hetero_predictability)),
            "median_absolute_predictability_change": float(np.median(np.abs(scalar_predictability-hetero_predictability))),
            "scalar_fraction_q_ge_0_5_and_snr_ge_1": float(np.mean((scalar_predictability>=0.5)&(scalar_snr>=1.0))),
            "heteroscedastic_fraction_q_ge_0_5_and_snr_ge_1": float(np.mean((hetero_predictability>=0.5)&(hetero_snr>=1.0))),
        },
        "lines": line_results,
        "decisions": {"uniform_confidence_across_extrapolated_support": False, "scalar_eta_is_physically_complete_noise_model": False, "posterior_covariance_is_physical_wideband_validation": False},
        "boundary": "Four selected phase-corrected public A-lines. Covariance is conditional on fitted IAA power and excludes power uncertainty, raw preprocessing, sample dispersion, chromatic transfer and independent physical outer-band truth."
    }
    (OUT/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    order=np.argsort(missing_indices)
    plt.figure(figsize=(8.6,5.3)); plt.plot(missing_indices[order],np.median(scalar_predictability,axis=0)[order],label="scalar eta"); plt.plot(missing_indices[order],np.median(hetero_predictability,axis=0)[order],label="heteroscedastic noise"); plt.xlabel("relative missing spectral-bin index"); plt.ylabel("median conditional predictability"); plt.ylim(-.03,1.03); plt.title("Official A-lines: confidence is nonuniform across extrapolated support"); plt.legend(); plt.tight_layout(); plt.savefig(OUT/"predictability_by_frequency.png",dpi=180); plt.close()
    plt.figure(figsize=(8.6,5.3)); plt.scatter(np.median(scalar_predictability,axis=0),np.median(hetero_predictability,axis=0),s=16); plt.plot([0,1],[0,1]); plt.xlabel("predictability with scalar eta"); plt.ylabel("predictability with heteroscedastic noise"); plt.title("Noise-model choice changes extrapolation confidence"); plt.tight_layout(); plt.savefig(OUT/"noise_model_comparison.png",dpi=180); plt.close()

if __name__=="__main__": main()
