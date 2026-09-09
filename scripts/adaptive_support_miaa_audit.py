from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from uncertainty_miaa_audit import (
    combine_results,
    conditional_miaa,
    normalized_noise_covariance,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "adaptive_support"
OUT.mkdir(parents=True, exist_ok=True)

L = 128
NG = 24
NFULL = 96
GIVEN = np.arange(-NG // 2, NG // 2)
FULL = np.arange(-NFULL // 2, NFULL // 2)
MISSING_MASK = ~np.isin(FULL, GIVEN)
MISSING = FULL[MISSING_MASK]
GIVEN_IN_FULL = np.flatnonzero(np.isin(FULL, GIVEN))
Z = np.arange(L)


def confidence_taper(predictability: np.ndarray, *, reject_below: float, accept_above: float) -> np.ndarray:
    if not 0 <= reject_below < accept_above <= 1:
        raise ValueError("invalid confidence thresholds")
    scaled = np.clip((np.asarray(predictability, float) - reject_below) / (accept_above - reject_below), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * scaled)


FG = np.exp(-1j * 2 * np.pi * GIVEN[:, None] * Z[None, :] / L)
FM = np.exp(-1j * 2 * np.pi * MISSING[:, None] * Z[None, :] / L)
FFULL = np.exp(-1j * 2 * np.pi * FULL[:, None] * Z[None, :] / L)


def iaa_power(observed: np.ndarray, noise_covariance: np.ndarray, iterations: int = 6) -> np.ndarray:
    observed = np.asarray(observed, np.complex128).reshape(-1)
    power = np.maximum(np.abs(FG.conj().T @ observed / NG) ** 2, 1e-14)
    for _ in range(iterations):
        rgg = (FG * power[None, :]) @ FG.conj().T + noise_covariance
        solved_y = np.linalg.solve(rgg, observed)
        solved_f = np.linalg.solve(rgg, FG)
        numerator = FG.conj().T @ solved_y
        denominator = np.sum(FG.conj() * solved_f, axis=0)
        denominator = np.where(np.abs(denominator) > 1e-14, denominator, 1e-14)
        estimate = numerator / denominator
        power = np.maximum(np.abs(estimate) ** 2, 1e-16)
    return power


def power_variants(power: np.ndarray) -> dict[str, np.ndarray]:
    maximum = max(float(np.max(power)), 1e-30)
    thresholded = power.copy()
    thresholded[thresholded < 0.01 * maximum] = 0.0
    return {
        "baseline": power,
        "shift_minus_1": np.roll(power, -1),
        "shift_plus_1": np.roll(power, 1),
        "smooth_0_8": gaussian_filter1d(power, 0.8, mode="wrap"),
        "threshold_1pct": thresholded,
    }


def embed_spectrum(given: np.ndarray, missing: np.ndarray) -> np.ndarray:
    output = np.zeros(NFULL, np.complex128)
    output[GIVEN_IN_FULL] = given
    output[MISSING_MASK] = missing
    return output


def reconstruct(spectrum: np.ndarray) -> np.ndarray:
    embedded = np.zeros(L, np.complex128)
    for index, value in zip(FULL, spectrum):
        embedded[int(index) % L] = value
    return np.fft.ifft(embedded) * L


def cscale(reference: np.ndarray, prediction: np.ndarray) -> complex:
    denominator = np.vdot(prediction, prediction)
    return np.vdot(prediction, reference) / denominator if abs(denominator) > 0 else 0j


def aligned_nmse(reference: np.ndarray, prediction: np.ndarray) -> float:
    scale = cscale(reference, prediction)
    return float(np.linalg.norm(reference - scale * prediction) ** 2 / max(np.linalg.norm(reference) ** 2, 1e-30))


def complex_correlation(reference: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.linalg.norm(reference) * np.linalg.norm(prediction)
    return float(abs(np.vdot(reference, prediction)) / denominator) if denominator > 0 else float("nan")


def circular_distance(a: int, b: int) -> int:
    distance = abs(a - b) % L
    return int(min(distance, L - distance))


def peak_metrics(profile: np.ndarray, true_positions: list[int]) -> dict[str, float | int]:
    intensity = np.abs(profile) ** 2
    intensity /= max(float(np.max(intensity)), 1e-30)
    peaks, _ = find_peaks(intensity, height=0.06, prominence=0.03, distance=1)
    peaks = peaks[np.argsort(intensity[peaks])[::-1]]
    used: set[int] = set()
    matches = []
    errors = []
    for truth in true_positions:
        candidates = [(circular_distance(int(peak), truth), index, int(peak)) for index, peak in enumerate(peaks) if index not in used]
        if not candidates:
            continue
        error, index, peak = min(candidates)
        if error <= 2:
            used.add(index)
            matches.append(peak)
            errors.append(error)
    false_peaks = sum(all(circular_distance(int(peak), truth) > 2 for truth in true_positions) and intensity[peak] > 0.15 for peak in peaks)
    mask = np.zeros(L, bool)
    for truth in true_positions:
        for offset in range(-2, 3):
            mask[(truth + offset) % L] = True
    artifact_energy = float(np.sum(intensity[~mask]) / max(np.sum(intensity), 1e-30))
    resolved_pair = False
    if len(true_positions) == 2 and len(matches) == 2:
        a, b = sorted(matches)
        segment = intensity[a:b + 1] if b - a <= L // 2 else np.r_[intensity[b:], intensity[:a + 1]]
        valley = float(np.min(segment))
        lower_peak = min(float(intensity[matches[0]]), float(intensity[matches[1]]))
        resolved_pair = valley <= 0.85 * lower_peak and false_peaks == 0
    return {
        "matched_fraction": len(matches) / max(len(true_positions), 1),
        "maximum_position_error_bins": float(max(errors)) if errors else float("inf"),
        "false_peak_count": int(false_peaks),
        "artifact_energy_fraction": artifact_energy,
        "resolved_pair": bool(resolved_pair),
    }


def scene(family: str, rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    reflectivity = np.zeros(L, np.complex128)
    if family == "single":
        positions = [64]; reflectivity[64] = 1.0
    elif family == "pair_close":
        positions = [60, 64]; reflectivity[60] = 1.0; reflectivity[64] = 0.6 * np.exp(1j * rng.uniform(-np.pi, np.pi))
    elif family == "pair_resolved":
        positions = [58, 66]; reflectivity[58] = 1.0; reflectivity[66] = 0.55 * np.exp(1j * rng.uniform(-np.pi, np.pi))
    elif family == "sparse":
        positions = sorted(rng.choice(np.arange(18, 110), size=5, replace=False).tolist())
        reflectivity[positions] = rng.uniform(0.25, 1.0, len(positions)) * np.exp(1j * rng.uniform(-np.pi, np.pi, len(positions)))
    elif family == "dense":
        positions = sorted(rng.choice(np.arange(18, 110), size=22, replace=False).tolist())
        reflectivity[positions] = rng.uniform(0.08, 0.7, len(positions)) * np.exp(1j * rng.uniform(-np.pi, np.pi, len(positions)))
    else:
        raise ValueError(family)
    return reflectivity, positions


def transfer(kind: str) -> np.ndarray:
    normalized = FULL / max(abs(FULL))
    if kind == "matched":
        return np.ones(NFULL, np.complex128)
    if kind == "quadratic_phase":
        return np.exp(1j * 2.0 * normalized**2)
    if kind == "combined":
        return np.exp(-0.65 * normalized**2) * np.exp(1j * 2.5 * normalized**2)
    raise ValueError(kind)


def confidence_weights(result) -> np.ndarray:
    phase_std = np.sqrt(result.posterior_variance / np.maximum(2.0 * np.abs(result.mean) ** 2, 1e-30))
    q_weight = confidence_taper(result.predictability, reject_below=0.25, accept_above=0.80)
    snr_weight = result.predicted_snr / (result.predicted_snr + 1.0)
    phase_weight = np.exp(-0.5 * np.minimum(phase_std, 8.0) ** 2)
    return gaussian_filter1d(np.clip(q_weight * snr_weight * phase_weight, 0.0, 1.0), 0.8, mode="nearest")


def fixed_cosine_weights() -> np.ndarray:
    distance = np.where(MISSING < GIVEN.min(), GIVEN.min() - MISSING, MISSING - GIVEN.max()).astype(float)
    normalized = distance / max(float(distance.max()), 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * normalized))


def run() -> dict[str, object]:
    rng = np.random.default_rng(20260909)
    families = ["single", "pair_close", "pair_resolved", "sparse", "dense"]
    transfers = ["matched", "quadratic_phase", "combined"]
    snrs = [20.0, 30.0, 40.0]
    trials = 18
    rows = []
    source_given = np.exp(-0.5 * (GIVEN / 9.5) ** 2)
    fixed_weight = fixed_cosine_weights()
    for family in families:
        for transfer_kind in transfers:
            h_full = transfer(transfer_kind); h_given = h_full[GIVEN_IN_FULL]
            for snr_db in snrs:
                for trial in range(trials):
                    reflectivity, positions = scene(family, rng)
                    true_full = (FFULL @ reflectivity) * h_full
                    clean_given = (FG @ reflectivity) * h_given
                    raw_signal_power = float(np.mean(np.abs(source_given * clean_given) ** 2))
                    raw_noise_variance = raw_signal_power / 10 ** (snr_db / 10)
                    raw_noise = np.sqrt(raw_noise_variance / 2) * (rng.standard_normal(NG) + 1j * rng.standard_normal(NG))
                    observed = clean_given + raw_noise / source_given
                    sigma = normalized_noise_covariance(source_given, raw_noise_variance=raw_noise_variance, relative_floor=0.03)
                    estimated_power = iaa_power(observed, sigma, iterations=6)
                    results = [conditional_miaa(observed, FG, FM, variant, sigma, max_condition_number=1e11) for variant in power_variants(estimated_power).values()]
                    baseline = results[0]; ensemble = combine_results(results); confidence = confidence_weights(ensemble)
                    methods = {
                        "measured_only": embed_spectrum(observed, np.zeros(MISSING.size, np.complex128)),
                        "fixed_full": embed_spectrum(observed, baseline.mean),
                        "fixed_cosine": embed_spectrum(observed, fixed_weight * baseline.mean),
                        "ensemble_full": embed_spectrum(observed, ensemble.mean),
                        "ensemble_confidence": embed_spectrum(observed, confidence * ensemble.mean),
                    }
                    true_profile = reconstruct(true_full)
                    for method, spectrum in methods.items():
                        peak = peak_metrics(reconstruct(spectrum), positions)
                        rows.append({
                            "family": family, "transfer": transfer_kind, "snr_db": snr_db, "trial": trial, "method": method,
                            "profile_aligned_nmse": aligned_nmse(true_profile, reconstruct(spectrum)),
                            "missing_complex_corr": complex_correlation(true_full[MISSING_MASK], spectrum[MISSING_MASK]),
                            "accepted_missing_fraction": float(np.mean(confidence >= 0.5)) if method == "ensemble_confidence" else 1.0 if method in ("fixed_full", "ensemble_full") else float(np.mean(fixed_weight >= 0.5)) if method == "fixed_cosine" else 0.0,
                            **peak,
                        })
    dataframe = pd.DataFrame(rows); dataframe.to_csv(OUT / "per_trial.csv", index=False)
    summary = []
    for (family, transfer_kind, snr_db, method), group in dataframe.groupby(["family", "transfer", "snr_db", "method"]):
        summary.append({
            "family": family, "transfer": transfer_kind, "snr_db": float(snr_db), "method": method, "n": int(len(group)),
            "median_profile_nmse": float(group["profile_aligned_nmse"].median()),
            "median_missing_corr": float(group["missing_complex_corr"].median()),
            "mean_false_peak_count": float(group["false_peak_count"].mean()),
            "median_artifact_energy_fraction": float(group["artifact_energy_fraction"].median()),
            "mean_matched_fraction": float(group["matched_fraction"].mean()),
            "pair_resolution_rate": float(group["resolved_pair"].mean()),
            "median_accepted_missing_fraction": float(group["accepted_missing_fraction"].median()),
        })
    summary_df = pd.DataFrame(summary); summary_df.to_csv(OUT / "summary.csv", index=False)
    mismatch = summary_df[summary_df["transfer"] != "matched"]; matched = summary_df[summary_df["transfer"] == "matched"]
    def aggregate(frame: pd.DataFrame, method: str, metric: str) -> float:
        return float(frame[frame["method"] == method][metric].median())
    decisions = {
        "matched_model_fixed_full_nmse": aggregate(matched, "fixed_full", "median_profile_nmse"),
        "matched_model_confidence_nmse": aggregate(matched, "ensemble_confidence", "median_profile_nmse"),
        "mismatch_fixed_full_nmse": aggregate(mismatch, "fixed_full", "median_profile_nmse"),
        "mismatch_confidence_nmse": aggregate(mismatch, "ensemble_confidence", "median_profile_nmse"),
        "mismatch_fixed_full_artifact_energy": aggregate(mismatch, "fixed_full", "median_artifact_energy_fraction"),
        "mismatch_confidence_artifact_energy": aggregate(mismatch, "ensemble_confidence", "median_artifact_energy_fraction"),
        "confidence_weighting_reduces_mismatch_nmse": aggregate(mismatch, "ensemble_confidence", "median_profile_nmse") < aggregate(mismatch, "fixed_full", "median_profile_nmse"),
        "confidence_weighting_reduces_mismatch_artifact_energy": aggregate(mismatch, "ensemble_confidence", "median_artifact_energy_fraction") < aggregate(mismatch, "fixed_full", "median_artifact_energy_fraction"),
        "confidence_weighting_is_free_resolution_gain": False,
        "confidence_weighting_replaces_wideband_truth": False,
    }
    report = {
        "schema": "adaptive-support-miaa-audit-v1", "status": "VERIFIED_SYNTHETIC",
        "configuration": {"depth_grid": L, "given_bins": NG, "full_bins": NFULL, "trials_per_cell": trials, "families": families, "transfers": transfers, "snrs_db": snrs, "confidence_rule": "model-ensemble predictability, predicted SNR and small-error phase variance; thresholds registered before the test"},
        "decisions": decisions, "summary": summary,
        "boundary": "Synthetic normalized-spectrum audit. The confidence ensemble is heuristic, and favorable results do not replace an independent physical wideband measurement.",
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(decisions, indent=2)); return report


if __name__ == "__main__":
    run()
