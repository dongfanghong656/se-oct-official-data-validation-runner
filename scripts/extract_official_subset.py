#!/usr/bin/env python3
"""Download Zenodo 7870795 TiO2 input and persist a small cited validation subset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import requests
from scipy.io import loadmat

URL = "https://zenodo.org/records/7870795/files/input_TiO2gelatin_004_phasecorrected.mat?download=1"
EXPECTED_MD5 = "d905784f950a6fadfce71e9efc6d2654"
COORDS = np.asarray([[12, 7], [232, 210], [482, 38], [1, 453]], dtype=np.int32)


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def download(path: Path) -> None:
    if path.exists() and digest(path, "md5") == EXPECTED_MD5:
        return
    part = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, 7):
        try:
            headers: dict[str, str] = {}
            mode = "wb"
            offset = part.stat().st_size if part.exists() else 0
            if offset:
                headers["Range"] = f"bytes={offset}-"
                mode = "ab"
            with requests.get(URL, stream=True, timeout=(30, 180), headers=headers) as r:
                if r.status_code == 200 and offset:
                    mode = "wb"
                r.raise_for_status()
                with part.open(mode) as f:
                    for chunk in r.iter_content(4 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            part.replace(path)
            actual = digest(path, "md5")
            if actual != EXPECTED_MD5:
                raise RuntimeError(f"MD5 mismatch: {actual}")
            return
        except Exception:
            if attempt == 6:
                raise
            time.sleep(min(60, 2**attempt))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cache = root / ".cache" / "input_TiO2gelatin_004_phasecorrected.mat"
    out = root / "reports" / "extracts"
    out.mkdir(parents=True, exist_ok=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    download(cache)
    volume = np.asarray(loadmat(cache, variable_names=["Cscan"])["Cscan"], dtype=np.complex64)
    if volume.shape != (200, 512, 512):
        raise RuntimeError(f"Unexpected Cscan shape {volume.shape}")

    z_lines = volume[:, COORDS[:, 0], COORDS[:, 1]]
    spectra = np.fft.fft(np.fft.ifftshift(z_lines, axes=0), axis=0).astype(np.complex64)

    rng = np.random.default_rng(7)
    random_flat = rng.choice(512 * 512, size=2048, replace=False)
    random_y, random_x = np.unravel_index(random_flat, (512, 512))
    sampled = volume[:, random_y, random_x]
    sampled_spectra = np.fft.fft(np.fft.ifftshift(sampled, axes=0), axis=0)
    source_profile = np.mean(np.abs(sampled_spectra), axis=1).astype(np.float32)

    y0, x0 = map(int, COORDS[1])
    half = 64
    x_start, x_stop = x0 - half, x0 + half + 1
    y_start, y_stop = y0 - half, y0 + half + 1
    x_strip = volume[:, y0, x_start:x_stop]
    y_strip = volume[:, y_start:y_stop, x0]

    subset = out / "official_tio2_selected_subset.npz"
    np.savez_compressed(
        subset,
        coordinates_yx=COORDS,
        z_lines=z_lines,
        full_spectra=spectra,
        source_profile=source_profile,
        x_strip=x_strip,
        x_strip_y=np.int32(y0),
        x_strip_x_start=np.int32(x_start),
        y_strip=y_strip,
        y_strip_x=np.int32(x0),
        y_strip_y_start=np.int32(y_start),
        axial_roi_um=np.float64(810.0),
        lateral_sampling_um=np.float64(0.44),
        center_wavelength_um=np.float64(0.510),
        refractive_index=np.float64(1.33),
        source_window_start=np.int32(49),
        source_window_stop=np.int32(177),
    )
    manifest = {
        "source_record": "10.5281/zenodo.7870795",
        "source_file": cache.name,
        "source_expected_md5": EXPECTED_MD5,
        "source_actual_md5": digest(cache, "md5"),
        "source_boundary": "Phase-corrected complex OCT C-scan, not raw camera interferograms.",
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "coordinates_yx": COORDS.tolist(),
        "subset_file": subset.name,
        "subset_sha256": digest(subset, "sha256"),
        "subset_bytes": subset.stat().st_size,
        "extraction": {
            "transform": "fft(ifftshift(z_line))",
            "source_profile": "mean abs spectrum over deterministic seed-7 sample of 2048 A-lines",
            "x_strip": [y0, x_start, x_stop],
            "y_strip": [y_start, y_stop, x0],
        },
        "use_constraint": "Public-data validation only; cite de Wit et al. and Zenodo record 7870795."
    }
    (out / "official_tio2_selected_subset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
