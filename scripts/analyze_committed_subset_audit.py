#!/usr/bin/env python3
"""Adversarial audit of exact-author RFIAA traversal and MIAA target support.

The analysis deliberately separates:
- the 128 input bins used to fit each A-line,
- the remaining 72 bins of the same deposited 200-bin representation,
- the completely unmeasured part of the constructed K-bin MIAA spectrum.

The 72-bin region is not treated as independent raw-camera wideband truth.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat


EPS = 1e-30


def complex_scale(ref: np.ndarray, pred: np.ndarray, weights: np.ndarray | None = None) -> complex:
    ref = np.asarray(ref).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    if weights is None:
        den = np.vdot(pred, pred)
        return np.vdot(pred, ref) / den if abs(den) > 0 else 0j
    w = np.asarray(weights, float).reshape(-1)
    den = np.sum(w * np.conj(pred) * pred)
    return np.sum(w * np.conj(pred) * ref) / den if abs(den) > 0 else 0j


def complex_metrics(
    ref: np.ndarray,
    pred: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    align: bool = True,
) -> dict[str, float]:
    ref = np.asarray(ref).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    if ref.size == 0:
        return {key: float("nan") for key in ("corr", "nmse", "phase_rmse_rad", "amplitude_nrmse")}
    w = None if weights is None else np.asarray(weights, float).reshape(-1)
    scale = complex_scale(ref, pred, w) if align else 1.0 + 0j
    estimate = scale * pred
    if w is None:
        ref_energy = float(np.sum(np.abs(ref) ** 2))
        est_energy = float(np.sum(np.abs(estimate) ** 2))
        cross = np.vdot(ref, estimate)
        err = float(np.sum(np.abs(ref - estimate) ** 2))
        amp_err = float(np.sum((np.abs(ref) - np.abs(estimate)) ** 2))
        phase = np.angle(ref * np.conj(estimate))
        phase_rmse = float(np.sqrt(np.mean(phase**2)))
    else:
        ref_energy = float(np.sum(w * np.abs(ref) ** 2))
        est_energy = float(np.sum(w * np.abs(estimate) ** 2))
        cross = np.sum(w * np.conj(ref) * estimate)
        err = float(np.sum(w * np.abs(ref - estimate) ** 2))
        amp_err = float(np.sum(w * (np.abs(ref) - np.abs(estimate)) ** 2))
        phase = np.angle(ref * np.conj(estimate))
        phase_rmse = float(np.sqrt(np.sum(w * phase**2) / max(float(np.sum(w)), EPS)))
    denom = np.sqrt(max(ref_energy * est_energy, 0.0))
    return {
        "corr": float(abs(cross) / denom) if denom > 0 else float("nan"),
        "nmse": float(err / max(ref_energy, EPS)),
        "phase_rmse_rad": phase_rmse,
        "amplitude_nrmse": float(np.sqrt(amp_err / max(ref_energy, EPS))),
        "scale_real": float(np.real(scale)),
        "scale_imag": float(np.imag(scale)),
    }


def quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray([float(v) for v in values if np.isfinite(v)], float)
    if array.size == 0:
        return {key: float("nan") for key in ("min", "p10", "median", "p90", "max")}
    return {
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def author_mapping(K: int, Nz: int, super_factor: int, ii_zero: np.ndarray) -> dict[str, int | str]:
    ii_first_matlab = int(ii_zero[0]) + 1
    shift_amount = int(round(Nz * ((super_factor - 1) / 2))) + ii_first_matlab - 1
    # Verified for super=4 against the deposited selected-A-line run:
    # reverse orientation, zero-based start 453.
    start_zero = K - shift_amount - ii_first_matlab
    return {
        "orientation": "reverse",
        "start_zero_based": int(start_zero),
        "shift_amount_matlab": int(shift_amount),
        "ii_first_matlab": ii_first_matlab,
    }


def orient(spectrum: np.ndarray) -> np.ndarray:
    return np.asarray(spectrum)[::-1, ...]


def masks_for_200(sk: np.ndarray, ii: np.ndarray) -> dict[str, np.ndarray]:
    n = sk.size
    index = np.arange(n)
    held = np.ones(n, bool)
    held[ii] = False
    normalized_source = np.abs(sk) / max(float(np.max(np.abs(sk))), EPS)
    near = held & (
        ((index >= max(0, ii[0] - 8)) & (index < ii[0]))
        | ((index > ii[-1]) & (index <= min(n - 1, ii[-1] + 8)))
    )
    return {
        "input_128": np.isin(index, ii),
        "heldout_all_72": held,
        "heldout_near_16": near,
        "heldout_far_56": held & ~near,
        "heldout_source_ge_5pct": held & (normalized_source >= 0.05),
        "heldout_source_lt_5pct": held & (normalized_source < 0.05),
        "heldout_left": index < ii[0],
        "heldout_right": index > ii[-1],
    }


def add_domain_metrics(
    row: dict[str, object],
    reference: np.ndarray,
    estimate: np.ndarray,
    sk: np.ndarray,
    mask: np.ndarray,
) -> None:
    ref = reference[mask]
    pred = estimate[mask]
    source_norm = np.abs(sk[mask]) / max(float(np.max(np.abs(sk))), EPS)
    # Use reference-only reliability weights. Prediction-dependent weights can
    # reward a model for suppressing bins where it is wrong.
    reference_signal_weight = source_norm**2 * np.abs(ref)**2
    reference_signal_weight = reference_signal_weight / max(
        float(np.max(reference_signal_weight)), EPS
    )
    domains = {
        "normalized": complex_metrics(ref, pred),
        "source_restored": complex_metrics(ref * sk[mask], pred * sk[mask]),
        "source_weighted": complex_metrics(ref, pred, source_norm**2),
        "signal_source_weighted": complex_metrics(
            ref,
            pred,
            reference_signal_weight,
        ),
    }
    for prefix, metrics in domains.items():
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value


def taper(length: int, edge: int) -> np.ndarray:
    if edge <= 0:
        return np.ones(length)
    edge = min(edge, length // 2)
    hanning = np.hanning(2 * edge)
    window = np.ones(length)
    window[:edge] = hanning[:edge]
    window[-edge:] = hanning[edge:]
    return window


def width_at_db(profile: np.ndarray, spacing_um: float, db: float) -> float:
    intensity = np.abs(np.asarray(profile).reshape(-1)) ** 2
    if intensity.size == 0 or not np.isfinite(intensity).all() or float(np.max(intensity)) <= 0:
        return float("nan")
    baseline = float(np.quantile(intensity, 0.01))
    intensity = np.maximum(intensity - baseline, 0.0)
    peak = int(np.argmax(intensity))
    threshold = float(intensity[peak]) * 10 ** (-db / 10)
    left = peak
    while left > 0 and intensity[left] >= threshold:
        left -= 1
    right = peak
    while right < intensity.size - 1 and intensity[right] >= threshold:
        right += 1
    if left == peak or right == peak:
        return float("nan")

    def crossing(a: int, b: int) -> float:
        if intensity[b] == intensity[a]:
            return float(a)
        return float(a + (threshold - intensity[a]) / (intensity[b] - intensity[a]))

    return (crossing(right - 1, right) - crossing(left, left + 1)) * spacing_um


def summarize_frame(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for group_values, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        record: dict[str, object] = {
            key: (int(value) if isinstance(value, (np.integer,)) else value)
            for key, value in zip(group_columns, group_values)
        }
        record["n_rows"] = int(len(group))
        record["n_lines"] = int(group["line"].nunique()) if "line" in group.columns else int(len(group))
        for metric in metric_columns:
            record[metric] = quantiles(group[metric].to_numpy(float))
        records.append(record)
    return records


def analyze_order(order_path: Path) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    data = loadmat(order_path, squeeze_me=True)
    Nz = int(data["Nz"])
    K = int(data["K"])
    super_factor = int(data["super"])
    ii = np.asarray(data["ii"], int).reshape(-1) - 1
    sk = np.asarray(data["sk"], float).reshape(-1)
    normalized_full = np.asarray(data["anC"], complex)
    line_scores = np.asarray(data["line_scores"], float).reshape(-1)
    x_indices = np.asarray(data["x_indices_zero_based"], int).reshape(-1)
    strong_index = int(data["strong_idx"]) - 1
    chunk_exact_ok = bool(int(data["chunk_exact_ok"]))
    chunk_error = str(data.get("chunk_exact_error", ""))

    spectra: dict[str, np.ndarray] = {
        "forward_whole_strip": np.asarray(data["spectra_forward"], complex),
        "reverse_whole_strip": np.asarray(data["spectra_reverse"], complex),
        "independent_q10": np.asarray(data["spectra_independent"], complex),
    }
    if chunk_exact_ok:
        spectra["forward_author_L4"] = np.asarray(data["spectra_chunk_forward"], complex)
        spectra["reverse_author_L4"] = np.asarray(data["spectra_chunk_reverse"], complex)

    mapping = author_mapping(K, Nz, super_factor, ii)
    start = int(mapping["start_zero_based"])
    oriented = {name: orient(value) for name, value in spectra.items()}
    blocks = {name: value[start : start + Nz, :] for name, value in oriented.items()}
    target = normalized_full - np.mean(normalized_full[ii, :], axis=0, keepdims=True)
    source_masks = masks_for_200(sk, ii)

    holdout_rows: list[dict[str, object]] = []
    for line in range(target.shape[1]):
        for mode, block in blocks.items():
            training_scale = complex_scale(target[ii, line], block[ii, line])
            estimate = training_scale * block[:, line]
            for region, mask in source_masks.items():
                row: dict[str, object] = {
                    "line": line,
                    "x_zero_based": int(x_indices[line]),
                    "mode": mode,
                    "region": region,
                    "n_bins": int(np.sum(mask)),
                    "signal_score": float(line_scores[line]),
                    "distance_from_strong_line": int(line - strong_index),
                    "position_class": "edge" if min(line, target.shape[1] - 1 - line) < 6 else "interior",
                    "side_of_strong_line": "before" if line < strong_index else ("after" if line > strong_index else "strong"),
                }
                add_domain_metrics(row, target[:, line], estimate, sk, mask)
                holdout_rows.append(row)

    embedded_mask = np.zeros(K, bool)
    embedded_mask[start : start + Nz] = True
    input_mask = np.zeros(K, bool)
    input_mask[start + ii] = True
    region_masks_K = {
        "input_128": input_mask,
        "embedded_200": embedded_mask,
        "heldout_72": embedded_mask & ~input_mask,
        "extrapolated_outside_200": ~embedded_mask,
        "full_K": np.ones(K, bool),
    }
    comparisons = [
        ("whole_forward_vs_reverse", "forward_whole_strip", "reverse_whole_strip"),
        ("whole_forward_vs_independent", "forward_whole_strip", "independent_q10"),
        ("whole_reverse_vs_independent", "reverse_whole_strip", "independent_q10"),
    ]
    if chunk_exact_ok:
        comparisons.extend([
            ("author_L4_forward_vs_reverse", "forward_author_L4", "reverse_author_L4"),
            ("author_L4_forward_vs_independent", "forward_author_L4", "independent_q10"),
            ("author_L4_reverse_vs_independent", "reverse_author_L4", "independent_q10"),
        ])

    pair_rows: list[dict[str, object]] = []
    for line in range(target.shape[1]):
        for label, ref_name, pred_name in comparisons:
            ref = oriented[ref_name][:, line]
            pred = oriented[pred_name][:, line]
            scale = complex_scale(ref[input_mask], pred[input_mask])
            aligned = scale * pred
            for region, mask in region_masks_K.items():
                row = {
                    "line": line,
                    "x_zero_based": int(x_indices[line]),
                    "comparison": label,
                    "region": region,
                    "n_bins": int(np.sum(mask)),
                    "signal_score": float(line_scores[line]),
                    "distance_from_strong_line": int(line - strong_index),
                    "position_class": "edge" if min(line, target.shape[1] - 1 - line) < 6 else "interior",
                    "side_of_strong_line": "before" if line < strong_index else ("after" if line > strong_index else "strong"),
                    **complex_metrics(ref[mask], aligned[mask], align=False),
                }
                pair_rows.append(row)

    holdout_df = pd.DataFrame(holdout_rows)
    pair_df = pd.DataFrame(pair_rows)
    holdout_metrics = [
        "normalized_corr", "normalized_nmse", "normalized_phase_rmse_rad",
        "source_restored_corr", "source_restored_nmse", "source_restored_phase_rmse_rad",
        "source_weighted_corr", "source_weighted_nmse",
        "signal_source_weighted_corr", "signal_source_weighted_nmse",
    ]
    pair_metrics = ["corr", "nmse", "phase_rmse_rad", "amplitude_nrmse"]
    result = {
        "experiment": "exact_author_RFIAA_traversal_order_on_committed_TiO2_strip",
        "boundary": {
            "input": "checksum-traced phase-corrected public TiO2 subset",
            "not_audit": "raw-camera k-linearization, dispersion compensation, and coverslip correction",
            "holdout": "72 same-pipeline weak-envelope bins; not independent physical wideband truth",
            "recursion": "whole-strip test plus exact oct_iaa_c1 L=4 path when supported by Octave",
        },
        "configuration": {
            "Nz": Nz,
            "K": K,
            "super": super_factor,
            "input_bins": int(ii.size),
            "strip_lines": int(target.shape[1]),
            "strong_line_within_strip_zero_based": strong_index,
            "strong_line_original_x_zero_based": int(x_indices[strong_index]),
            "mapping": mapping,
            "chunk_exact_ok": chunk_exact_ok,
            "chunk_exact_error": chunk_error,
        },
        "holdout_summary": summarize_frame(
            holdout_df, ["mode", "region"], holdout_metrics
        ),
        "pair_summary": summarize_frame(
            pair_df, ["comparison", "region", "position_class"], pair_metrics
        ),
        "pair_side_summary": summarize_frame(
            pair_df[pair_df["region"].isin(["extrapolated_outside_200", "heldout_72"])],
            ["comparison", "region", "side_of_strong_line"],
            pair_metrics,
        ),
        "decision_rules": [
            "Input-128 agreement is necessary but does not adjudicate extrapolation.",
            "Forward/reverse disagreement after input-bin alignment is traversal-history dependence.",
            "Disagreement concentrated outside the embedded 200 bins indicates that the speed-up changes model extrapolation more than measured-data fit.",
            "FWHM is not used to decide traversal invariance.",
        ],
    }
    return result, holdout_df, pair_df


def analyze_superfactor(super_path: Path) -> tuple[dict[str, object], pd.DataFrame]:
    data = loadmat(super_path, squeeze_me=True)
    Nz = int(data["Nz"])
    supers = np.asarray(data["supers"], int).reshape(-1)
    ii = np.asarray(data["ii"], int).reshape(-1) - 1
    sk = np.asarray(data["sk"], float).reshape(-1)
    normalized_full = np.asarray(data["anC4"], complex)
    spectra_cube = np.asarray(data["spectra_super"], complex)
    source_masks = masks_for_200(sk, ii)
    target = normalized_full - np.mean(normalized_full[ii, :], axis=0, keepdims=True)
    rows: list[dict[str, object]] = []
    registrations = []

    for super_index, super_factor in enumerate(supers):
        K = Nz * int(super_factor)
        spectra = spectra_cube[:K, :, super_index]
        mapping = author_mapping(K, Nz, int(super_factor), ii)
        start = int(mapping["start_zero_based"])
        oriented = orient(spectra)
        block = oriented[start : start + Nz, :]
        registrations.append({"super": int(super_factor), "K": K, **mapping})
        windows = {
            "none": np.ones(K),
            "fixed_100_each_edge": taper(K, 100),
            "proportional_12p5pct_each_edge": taper(K, int(round(K / 8))),
        }
        spacing_um = 810.0 / K
        for line in range(target.shape[1]):
            scale = complex_scale(target[ii, line], block[ii, line])
            estimate = scale * block[:, line]
            widths: dict[str, float] = {}
            for window_name, window in windows.items():
                profile = np.fft.ifft(spectra[:, line] * window)
                for db in (3, 10, 20, 40):
                    widths[f"{window_name}_w{db}db_um"] = width_at_db(profile, spacing_um, db)
            for region, mask in source_masks.items():
                row: dict[str, object] = {
                    "super": int(super_factor),
                    "K": K,
                    "support_to_input_ratio": float(K / ii.size),
                    "line": line,
                    "region": region,
                    "n_bins": int(np.sum(mask)),
                    **widths,
                }
                add_domain_metrics(row, target[:, line], estimate, sk, mask)
                rows.append(row)

    frame = pd.DataFrame(rows)
    metrics = [
        "normalized_corr", "normalized_nmse",
        "source_restored_corr", "source_restored_nmse",
        "source_weighted_corr", "source_weighted_nmse",
        "signal_source_weighted_corr", "signal_source_weighted_nmse",
        "none_w3db_um", "none_w20db_um", "none_w40db_um",
        "fixed_100_each_edge_w3db_um", "fixed_100_each_edge_w20db_um", "fixed_100_each_edge_w40db_um",
        "proportional_12p5pct_each_edge_w3db_um",
        "proportional_12p5pct_each_edge_w20db_um",
        "proportional_12p5pct_each_edge_w40db_um",
    ]
    summary = summarize_frame(frame, ["super", "region"], metrics)

    # A conservative deterministic selection rule. This is an internal diagnostic, not physical validation.
    strong_region = frame[frame["region"] == "heldout_source_ge_5pct"]
    candidates: list[dict[str, float | int]] = []
    for super_factor, group in strong_region.groupby("super"):
        candidates.append({
            "super": int(super_factor),
            "source_restored_nmse_median": float(np.median(group["source_restored_nmse"])),
            "signal_source_weighted_nmse_median": float(np.median(group["signal_source_weighted_nmse"])),
            "fixed_taper_fwhm_median_um": float(np.median(group["fixed_100_each_edge_w3db_um"])),
        })
    best_restored = min(candidates, key=lambda item: item["source_restored_nmse_median"])
    best_signal = min(candidates, key=lambda item: item["signal_source_weighted_nmse_median"])
    selection = {
        "candidates": candidates,
        "best_super_source_restored": int(best_restored["super"]),
        "best_super_signal_source_weighted": int(best_signal["super"]),
        "super4_selected_by_both_metrics": bool(
            int(best_restored["super"]) == 4 and int(best_signal["super"]) == 4
        ),
        "scientific_status": "INTERNAL_SUPPORT_DIAGNOSTIC_ONLY",
    }

    result = {
        "experiment": "exact_author_MIAA_target_support_and_window_on_committed_TiO2_lines",
        "boundary": {
            "input": "four checksum-traced phase-corrected TiO2 A-lines",
            "holdout": "72 same-pipeline bins, not independent wider-band camera truth",
            "selection": "no parameter is fit to the holdout region; mapping is derived from author indexing",
        },
        "registrations": registrations,
        "summary": summary,
        "support_selection_diagnostic": selection,
        "decision_rules": [
            "Peak narrowing with larger super is not measured bandwidth.",
            "A support choice should improve source-restored and signal-weighted heldout prediction, not FWHM alone.",
            "Fixed-100 and proportional tapers are separated because the window fraction changes with K.",
            "Even a unique same-pipeline heldout optimum would remain weaker than an independent physical wideband holdout.",
        ],
    }
    return result, frame


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: analyze_committed_subset_audit.py ORDER_MAT SUPERFACTOR_MAT OUTPUT_DIR")
    output = Path(sys.argv[3])
    output.mkdir(parents=True, exist_ok=True)
    order_result, order_holdout, order_pairs = analyze_order(Path(sys.argv[1]))
    super_result, super_frame = analyze_superfactor(Path(sys.argv[2]))
    order_holdout.to_csv(output / "order_holdout_per_line.csv", index=False)
    order_pairs.to_csv(output / "order_pair_per_line.csv", index=False)
    super_frame.to_csv(output / "superfactor_per_line.csv", index=False)
    (output / "order_metrics.json").write_text(json.dumps(order_result, indent=2), encoding="utf-8")
    (output / "superfactor_metrics.json").write_text(json.dumps(super_result, indent=2), encoding="utf-8")
    combined = {
        "schema_version": "1.0",
        "order_audit": order_result,
        "superfactor_audit": super_result,
    }
    (output / "metrics.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps({
        "order_configuration": order_result["configuration"],
        "order_pair_summary": order_result["pair_summary"],
        "superfactor_selection": super_result["support_selection_diagnostic"],
    }, indent=2))


if __name__ == "__main__":
    main()
