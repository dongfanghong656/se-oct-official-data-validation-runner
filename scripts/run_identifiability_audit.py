#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seoct_uncertainty import (
    single_reflector_localization_crlb,
    two_reflector_separation_crlb,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "identifiability_audit"
OUT.mkdir(parents=True, exist_ok=True)


def conventional_psf_fwhm(k: np.ndarray) -> float:
    z = np.linspace(-20.0, 20.0, 200001)
    field = np.exp(1j * 2.0 * k[:, None] * z[None, :]).sum(axis=0)
    intensity = np.abs(field) ** 2
    intensity /= intensity.max()
    center = int(np.argmax(intensity))
    left = center
    right = center
    while left > 0 and intensity[left] >= 0.5:
        left -= 1
    while right < intensity.size - 1 and intensity[right] >= 0.5:
        right += 1
    def crossing(i0: int, i1: int) -> float:
        y0, y1 = intensity[i0], intensity[i1]
        if y1 == y0:
            return float(z[i0])
        return float(z[i0] + (0.5 - y0) / (y1 - y0) * (z[i1] - z[i0]))
    return crossing(right - 1, right) - crossing(left, left + 1)


def main() -> None:
    k = np.linspace(10.0, 11.0, 64)
    psf_fwhm = conventional_psf_fwhm(k)
    single = []
    for snr_db in (0, 5, 10, 15, 20, 25, 30, 35, 40, 50):
        result = single_reflector_localization_crlb(k, 1.0, 0.4, np.eye(k.size) * 10 ** (-snr_db / 10))
        single.append({
            "snr_db_per_sample": snr_db,
            "localization_standard_deviation": result.z_standard_deviation,
            "localization_std_over_conventional_psf_fwhm": result.z_standard_deviation / psf_fwhm,
            "fisher_condition_number": result.condition_number,
            "effective_rank": result.effective_rank,
            "identifiable": result.identifiable,
        })

    separations = np.r_[0.0, np.linspace(0.002, 3.0 * psf_fwhm, 140)]
    two = []
    for phase in (0.0, np.pi / 2.0, np.pi):
        for ratio in (1.0, 0.5):
            for separation in separations:
                result = two_reflector_separation_crlb(
                    k,
                    1.0,
                    ratio * np.exp(1j * phase),
                    0.4,
                    float(separation),
                    np.eye(k.size) * 1e-3,
                )
                two.append({
                    "relative_phase_rad": float(phase),
                    "amplitude_ratio": ratio,
                    "separation": float(separation),
                    "separation_over_conventional_psf_fwhm": float(separation / psf_fwhm),
                    "separation_standard_deviation": result.separation_standard_deviation,
                    "separation_std_over_true_separation": (
                        float(result.separation_standard_deviation / separation)
                        if separation > 0 and result.identifiable
                        else float("inf")
                    ),
                    "fisher_condition_number": result.condition_number,
                    "effective_rank": result.effective_rank,
                    "identifiable": result.identifiable,
                })

    report = {
        "schema": "se-oct-identifiability-audit-v2",
        "status": "VERIFIED_SYNTHETIC",
        "conventional_psf_fwhm_normalized_units": psf_fwhm,
        "single_reflector": single,
        "two_reflector": two,
        "decisions": {
            "single_reflector_localization_is_same_as_two_reflector_resolution": False,
            "rank_deficient_fisher_should_return_finite_pseudoinverse_bound": False,
            "single_point_fwhm_proves_universal_resolution": False,
            "crlb_is_physical_wideband_validation": False,
        },
        "boundary": "Optimistic proper-complex Gaussian CRLB with correct target count and model; excludes target-number selection, false peaks, multiple scattering, chromatic transfer and physical wideband truth.",
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    plt.figure(figsize=(8.4, 5.2))
    plt.semilogy(
        [row["snr_db_per_sample"] for row in single],
        [row["localization_std_over_conventional_psf_fwhm"] for row in single],
        marker="o",
    )
    plt.xlabel("per-sample SNR (dB)")
    plt.ylabel("localization std / DFT PSF FWHM")
    plt.title("Single-target localization can be finer than the Fourier main lobe")
    plt.tight_layout()
    plt.savefig(OUT / "single_target_localization_crlb.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.4, 5.2))
    for phase in (0.0, np.pi / 2.0, np.pi):
        rows = [
            row for row in two
            if row["amplitude_ratio"] == 1.0
            and row["relative_phase_rad"] == float(phase)
            and row["identifiable"]
            and np.isfinite(row["separation_std_over_true_separation"])
        ]
        plt.semilogy(
            [row["separation_over_conventional_psf_fwhm"] for row in rows],
            [row["separation_std_over_true_separation"] for row in rows],
            label=f"phase={phase / np.pi:.1f}pi",
        )
    plt.xlabel("true separation / conventional DFT PSF FWHM")
    plt.ylabel("CRLB separation std / true separation")
    plt.title("Two-target separation loses identifiability at small spacing")
    plt.ylim(1e-3, 1e4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "two_target_separation_crlb.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
