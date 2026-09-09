#!/usr/bin/env python3
"""Uncertainty-aware audit for phase-preserving MIAA.

Reports within-model conditional covariance, heteroscedastic noise after source
normalization, between-model disagreement, phase-confidence gates, synthetic
calibration, and an audit on the committed exact author A-lines.

Posterior covariance is conditional on the fitted model; it is not independent
physical wideband validation. Public author inputs are already phase corrected.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class ConditionalResult:
    mean: np.ndarray
    covariance: np.ndarray
    posterior_variance: np.ndarray
    prior_variance: np.ndarray
    predictability: np.ndarray
    predicted_snr: np.ndarray
    condition_number: float
    jitter_used: float


def hermitian(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conj().T)


def fourier(indices: np.ndarray, object_grid: int) -> np.ndarray:
    z = np.arange(object_grid)
    return np.exp(-1j * 2 * np.pi * np.asarray(indices)[:, None] * z[None, :] / object_grid)


def normalized_noise_covariance(source_spectrum: np.ndarray, raw_noise_variance: float | np.ndarray = 1.0, *, relative_floor: float = 1e-3) -> np.ndarray:
    """Propagate raw-domain noise through y_norm = y_raw / S."""
    source = np.asarray(source_spectrum, np.complex128).reshape(-1)
    variance = np.asarray(raw_noise_variance, float)
    if variance.ndim == 0:
        variance = np.full(source.size, float(variance))
    variance = variance.reshape(-1)
    if variance.size != source.size or np.any(~np.isfinite(variance)) or np.any(variance < 0):
        raise ValueError("invalid raw_noise_variance")
    floor = relative_floor * max(float(np.max(np.abs(source))), 1e-30)
    safe = source.copy()
    weak = np.abs(safe) < floor
    safe[weak] = floor * np.exp(1j * np.angle(safe[weak]))
    inv = 1.0 / safe
    return hermitian(inv[:, None] * np.diag(variance.astype(np.complex128)) * inv.conj()[None, :])


def conditional_miaa(y_given: np.ndarray, fourier_given: np.ndarray, fourier_missing: np.ndarray, reflectivity_power: np.ndarray, noise_covariance_given: np.ndarray, *, max_condition_number: float = 1e12, initial_relative_jitter: float = 1e-12) -> ConditionalResult:
    """Compute the MMSE conditional mean and Schur-complement covariance."""
    y = np.asarray(y_given, np.complex128).reshape(-1)
    fg = np.asarray(fourier_given, np.complex128)
    fm = np.asarray(fourier_missing, np.complex128)
    power = np.asarray(reflectivity_power, float).reshape(-1)
    sigma = hermitian(np.asarray(noise_covariance_given, np.complex128))
    if fg.ndim != 2 or fm.ndim != 2 or fg.shape[0] != y.size or fg.shape[1] != fm.shape[1]:
        raise ValueError("Fourier matrix shape mismatch")
    if power.size != fg.shape[1] or np.any(power < 0):
        raise ValueError("reflectivity power mismatch")
    if sigma.shape != (y.size, y.size):
        raise ValueError("noise covariance shape mismatch")
    if not all(np.all(np.isfinite(v)) for v in (y, fg, fm, power, sigma)):
        raise ValueError("nonfinite input")

    weighted_g = fg * power[None, :]
    weighted_m = fm * power[None, :]
    rgg = hermitian(weighted_g @ fg.conj().T + sigma)
    rmg = weighted_m @ fg.conj().T
    rmm = hermitian(weighted_m @ fm.conj().T)
    scale = max(float(np.real(np.trace(rgg))) / max(rgg.shape[0], 1), 1e-30)
    jitter = max(initial_relative_jitter * scale, 0.0)
    eye = np.eye(rgg.shape[0], dtype=np.complex128)
    solved_y = solved_cross = None
    condition = float("inf")
    for _ in range(12):
        regularized = rgg + jitter * eye
        condition = float(np.linalg.cond(regularized))
        if np.isfinite(condition) and condition <= max_condition_number:
            try:
                solved_y = np.linalg.solve(regularized, y)
                solved_cross = np.linalg.solve(regularized, rmg.conj().T)
                break
            except np.linalg.LinAlgError:
                pass
        jitter = max(jitter * 10.0, 1e-15 * scale)
    if solved_y is None or solved_cross is None:
        raise np.linalg.LinAlgError("stable conditional solve unavailable")

    mean = rmg @ solved_y
    covariance = hermitian(rmm - rmg @ solved_cross)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    reference = max(float(np.max(np.abs(eigvals))), 1.0)
    if float(np.min(eigvals)) < -1e-7 * reference:
        raise np.linalg.LinAlgError("conditional covariance materially indefinite")
    eigvals = np.maximum(eigvals, 0.0)
    covariance = hermitian((eigvecs * eigvals[None, :]) @ eigvecs.conj().T)
    posterior = np.maximum(np.real(np.diag(covariance)), 0.0)
    prior = np.maximum(np.real(np.diag(rmm)), 0.0)
    predictability = np.zeros_like(prior)
    valid = prior > 1e-30
    predictability[valid] = 1.0 - posterior[valid] / prior[valid]
    predictability = np.clip(predictability, 0.0, 1.0)
    predicted_snr = np.abs(mean) ** 2 / np.maximum(posterior, 1e-30)
    return ConditionalResult(mean, covariance, posterior, prior, predictability, predicted_snr, condition, float(jitter))


def combine_results(results: Iterable[ConditionalResult]) -> ConditionalResult:
    """Add between-model disagreement using the law of total covariance."""
    results = list(results)
    if not results:
        raise ValueError("empty model ensemble")
    size = results[0].mean.size
    if any(result.mean.size != size for result in results):
        raise ValueError("ensemble shape mismatch")
    means = np.stack([result.mean for result in results])
    mean = np.mean(means, axis=0)
    covariance = np.zeros((size, size), np.complex128)
    prior = np.zeros(size, float)
    for result in results:
        delta = result.mean - mean
        covariance += (result.covariance + np.outer(delta, delta.conj())) / len(results)
        prior += result.prior_variance / len(results)
    covariance = hermitian(covariance)
    posterior = np.maximum(np.real(np.diag(covariance)), 0.0)
    predictability = np.zeros_like(prior)
    valid = prior > 1e-30
    predictability[valid] = 1.0 - posterior[valid] / prior[valid]
    predictability = np.clip(predictability, 0.0, 1.0)
    predicted_snr = np.abs(mean) ** 2 / np.maximum(posterior, 1e-30)
    return ConditionalResult(mean, covariance, posterior, prior, predictability, predicted_snr, max(r.condition_number for r in results), max(r.jitter_used for r in results))


def phase_standard_deviation(result: ConditionalResult) -> np.ndarray:
    amplitude_squared = np.abs(result.mean) ** 2
    output = np.full(result.mean.size, np.inf)
    valid = amplitude_squared > 1e-30
    output[valid] = np.sqrt(result.posterior_variance[valid] / np.maximum(2.0 * amplitude_squared[valid], 1e-30))
    return output


def coherent_mask(result: ConditionalResult, *, min_predictability: float = 0.5, min_predicted_snr: float = 1.0, max_phase_std_rad: float = 0.3) -> np.ndarray:
    return ((result.predictability >= min_predictability) & (result.predicted_snr >= min_predicted_snr) & (phase_standard_deviation(result) <= max_phase_std_rad))


def power_variants(power: np.ndarray) -> dict[str, np.ndarray]:
    power = np.maximum(np.asarray(power, float), 0.0)
    maximum = max(float(np.max(power)), 1e-30)
    thresholded = power.copy()
    thresholded[thresholded < 0.01 * maximum] = 0.0
    return {
        "baseline": power,
        "shift_minus_1": np.roll(power, -1),
        "shift_plus_1": np.roll(power, 1),
        "smoothed_sigma_1": gaussian_filter1d(power, 1.0, mode="wrap"),
        "threshold_1pct": thresholded,
    }


def self_test() -> dict[str, object]:
    object_grid = 32
    fg = fourier(np.arange(-4, 4), object_grid)
    fm = fourier(np.arange(4, 12), object_grid)
    power = np.zeros(object_grid)
    power[[7, 13]] = [1.0, 0.4]
    y = fg @ np.sqrt(power)
    low = conditional_miaa(y, fg, fm, power, np.eye(8) * 1e-4)
    high = conditional_miaa(y, fg, fm, power, np.eye(8) * 1e-1)
    assert np.min(np.linalg.eigvalsh(low.covariance)) >= -1e-9
    assert np.all(low.posterior_variance <= low.prior_variance + 1e-9)
    assert float(np.mean(low.predictability)) > float(np.mean(high.predictability))
    ensemble = combine_results([low, ConditionalResult(low.mean * np.exp(0.4j), low.covariance, low.posterior_variance, low.prior_variance, low.predictability, low.predicted_snr, low.condition_number, low.jitter_used)])
    assert np.all(ensemble.posterior_variance >= low.posterior_variance - 1e-12)
    np.testing.assert_allclose(normalized_noise_covariance(np.array([1.0, 0.5, 0.25]), np.array([2.0, 2.0, 2.0])), np.diag([2.0, 8.0, 32.0]))
    return {"status": "VERIFIED", "tests": ["conditional covariance PSD", "posterior <= prior", "higher noise reduces predictability", "ensemble adds model variance", "source-normalized noise propagation"]}


def calibration_metrics(errors_squared: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    variance = np.maximum(variance, 1e-30)
    empirical_mse = np.mean(errors_squared, axis=0)
    ratio = errors_squared / variance[None, :]
    return {
        "median_empirical_mse_over_reported_variance": float(np.median(empirical_mse / variance)),
        "q90_empirical_mse_over_reported_variance": float(np.quantile(empirical_mse / variance, 0.90)),
        "nominal_95pct_complex_error_coverage": float(np.mean(ratio <= -math.log(0.05))),
    }


def synthetic_stress(output_dir: Path) -> dict[str, object]:
    rng = np.random.default_rng(20260909)
    object_grid = 128
    given_indices = np.arange(-12, 12)
    missing_indices = np.concatenate([np.arange(-48, -12), np.arange(12, 48)])
    fg = fourier(given_indices, object_grid)
    fm = fourier(missing_indices, object_grid)
    reflectivity = np.zeros(object_grid, np.complex128)
    reflectivity[31] = 1.0
    reflectivity[53] = 0.45 * np.exp(0.9j)
    reflectivity[82] = 0.24 * np.exp(-1.2j)
    true_power = np.abs(reflectivity) ** 2
    y_given = fg @ reflectivity
    y_missing = fm @ reflectivity
    noise_variance = 4e-4
    sigma = np.eye(given_indices.size) * noise_variance
    variants = power_variants(true_power)
    variants["miss_weak_reflector"] = true_power.copy(); variants["miss_weak_reflector"][82] = 0.0
    variants["spurious_reflector"] = true_power.copy(); variants["spurious_reflector"][69] += 0.22
    trial_errors = {name: [] for name in variants}
    reported_variance = {}
    ensemble_errors = []; ensemble_variances = []
    trials = 600
    for _ in range(trials):
        noise = np.sqrt(noise_variance / 2) * (rng.standard_normal(given_indices.size) + 1j * rng.standard_normal(given_indices.size))
        observed = y_given + noise
        results = []
        for name, power in variants.items():
            result = conditional_miaa(observed, fg, fm, power, sigma)
            trial_errors[name].append(np.abs(result.mean - y_missing) ** 2)
            reported_variance.setdefault(name, result.posterior_variance)
            results.append(result)
        ensemble = combine_results(results)
        ensemble_errors.append(np.abs(ensemble.mean - y_missing) ** 2)
        ensemble_variances.append(ensemble.posterior_variance)
    rows = []
    for name in variants:
        row = {"model": name}; row.update(calibration_metrics(np.asarray(trial_errors[name]), reported_variance[name])); rows.append(row)
    ensemble_row = {"model": "equal_weight_model_ensemble"}; ensemble_row.update(calibration_metrics(np.asarray(ensemble_errors), np.mean(np.asarray(ensemble_variances), axis=0))); rows.append(ensemble_row)
    report = {"schema": "miaa-model-misspecification-stress-v2", "status": "VERIFIED_SYNTHETIC", "trials": trials, "models": rows, "decisions": {"fixed_model_covariance_includes_model_error": False, "between_model_covariance_should_be_reported": True, "ensemble_replaces_physical_wideband_validation": False}, "boundary": "The registered ensemble is incomplete; absent physical failure modes can still produce undercoverage."}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def as_2d(value: np.ndarray, rows: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1: array = array[:, None]
    if rows is not None and array.shape[0] != rows and array.shape[1] == rows: array = array.T
    return array


def official_audit(source_path: Path, output_dir: Path) -> dict[str, object]:
    data = loadmat(source_path, squeeze_me=True)
    object_grid = int(np.asarray(data["K"]).item())
    y = as_2d(np.asarray(data["demod_spectrum"], np.complex128))
    reflectivity = as_2d(np.asarray(data["rfiaa_profile"], np.complex128), object_grid)
    pe = as_2d(np.asarray(data["pe_first"], float))
    source = np.asarray(data["sk"], float).reshape(-1)
    ii = np.asarray(data["ii"], int).reshape(-1) - 1
    if y.shape[0] != ii.size: y = y.T
    if pe.shape[1] != y.shape[1]: pe = pe.T
    given_indices = np.arange(y.shape[0])
    all_target = np.arange(-336, 464)
    missing_all = all_target[(all_target < 0) | (all_target >= y.shape[0])]
    take = np.unique(np.rint(np.linspace(0, missing_all.size - 1, 240)).astype(int))
    missing_indices = missing_all[take]
    distance = np.where(missing_indices < 0, -missing_indices, missing_indices - (y.shape[0] - 1))
    fg = fourier(given_indices, object_grid); fm = fourier(missing_indices, object_grid)
    source_given = np.abs(source[ii]); source_given /= max(float(np.max(source_given)), 1e-30)
    lines = []; model_fractions = []; baseline_q = []; ensemble_q = []
    for line in range(y.shape[1]):
        power = np.abs(reflectivity[:, line]) ** 2
        eta = max(float(pe[-1, line]), 1e-14)
        sigma = normalized_noise_covariance(source_given, eta * float(np.median(source_given ** 2)), relative_floor=0.03)
        results = [conditional_miaa(y[:, line], fg, fm, variant, sigma) for variant in power_variants(power).values()]
        baseline = results[0]; ensemble = combine_results(results)
        within = np.mean(np.stack([result.posterior_variance for result in results]), axis=0)
        model_fraction = np.clip(1.0 - within / np.maximum(ensemble.posterior_variance, 1e-30), 0.0, 1.0)
        model_fractions.append(model_fraction); baseline_q.append(baseline.predictability); ensemble_q.append(ensemble.predictability)
        bm = coherent_mask(baseline); em = coherent_mask(ensemble)
        lines.append({"line": line, "eta": eta, "baseline_median_predictability": float(np.median(baseline.predictability)), "ensemble_median_predictability": float(np.median(ensemble.predictability)), "median_model_variance_fraction": float(np.median(model_fraction)), "baseline_coherent_fraction": float(np.mean(bm)), "ensemble_coherent_fraction": float(np.mean(em)), "baseline_maximum_coherent_distance_bins": float(np.max(distance[bm]) if np.any(bm) else 0.0), "ensemble_maximum_coherent_distance_bins": float(np.max(distance[em]) if np.any(em) else 0.0)})
    report = {"schema": "official-phase-corrected-uncertainty-audit-v3", "status": "VERIFIED_CONDITIONAL_MODEL_AUDIT", "source": {"path": source_path.as_posix(), "n_lines": int(y.shape[1]), "given_bins": int(y.shape[0]), "object_grid": object_grid, "sampled_missing_bins": int(missing_indices.size)}, "aggregate": {"baseline_median_predictability": float(np.median(np.asarray(baseline_q))), "ensemble_median_predictability": float(np.median(np.asarray(ensemble_q))), "median_model_variance_fraction": float(np.median(np.asarray(model_fractions)))}, "lines": lines, "decisions": {"fixed_model_predictability_includes_model_error": False, "confidence_is_uniform_across_support": False, "posterior_covariance_is_physical_wideband_validation": False}, "boundary": "Four selected, already phase-corrected A-lines; heuristic power variants give only a lower-bound model-disagreement diagnostic."}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("self-test", "synthetic", "official", "all"), default="all")
    parser.add_argument("--author-alines", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"schema": "uncertainty-miaa-audit-bundle-v1", "status": "VERIFIED"}
    if args.mode in ("self-test", "all"): report["self_test"] = self_test()
    if args.mode in ("synthetic", "all"): report["synthetic"] = synthetic_stress(args.output / "synthetic")
    if args.mode in ("official", "all"):
        if args.author_alines is None: raise SystemExit("--author-alines is required for official/all")
        report["official"] = official_audit(args.author_alines, args.output / "official")
    (args.output / "combined_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
