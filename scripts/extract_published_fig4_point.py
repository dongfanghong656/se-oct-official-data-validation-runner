#!/usr/bin/env python3
"""Extract and quantify the Figure-4 out-of-focus point from authors' processed full-volume outputs.

This validates the published final outputs without reimplementing ISAM. It does not validate the
unmeasured MIAA spectrum or the unavailable raw-camera preprocessing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.optimize import least_squares

RECORD = "https://zenodo.org/records/7870795/files"
FILES = {
    "DFT_ISAM": ("exp_pointscat_image_DFT_ISAM.mat", "206e900fe93fb8306cc510a839d23e05"),
    "MIAA_ISAM": ("exp_pointscat_image_MIAA_ISAM.mat", "1fb312bcd855ea437d996629231b1e56"),
}
# Figure-3/4 white-arrow point: x=0.1985 mm, y=0.0975 mm, final full z index 461 (MATLAB 1-based).
# Saved amplitude files contain full-image z indices 200:1000, so zero-based saved z = 461-200 = 261.
EXPECTED_ZXY0 = np.asarray([261, 451, 221], dtype=int)
Z_HALF = 20
XY_HALF = 24
DX_UM = 225.0 / 512.0
LAMBDA_MIN = 5.011840350737296e-7
LAMBDA_MAX = 5.25482346025955e-7
N_REF = 1.33
NISAM = 1029
KMIN = 2 * np.pi / LAMBDA_MAX * N_REF
KMAX = 2 * np.pi / LAMBDA_MIN * N_REF
ZMAX_UM = np.pi * (200 - 1) / (KMAX - KMIN) * 1e6
DZ_UM = ZMAX_UM / (NISAM - 1)


def digest(path: Path, algorithm: str = "md5") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, expected_md5: str, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    if path.exists() and digest(path) == expected_md5:
        return path
    part = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, 8):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            mode = "ab" if offset else "wb"
            with requests.get(f"{RECORD}/{name}?download=1", stream=True, timeout=(30, 300), headers=headers) as r:
                if r.status_code == 200 and offset:
                    mode = "wb"
                r.raise_for_status()
                with part.open(mode) as f:
                    for chunk in r.iter_content(8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            part.replace(path)
            actual = digest(path)
            if actual != expected_md5:
                raise RuntimeError(f"MD5 mismatch for {name}: {actual}")
            return path
        except Exception:
            if attempt == 7:
                raise
            time.sleep(min(120, 2**attempt))
    raise AssertionError("unreachable")


def find_image_dataset(h5: h5py.File) -> h5py.Dataset:
    candidates: list[h5py.Dataset] = []
    def visit(_name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3 and np.issubdtype(obj.dtype, np.number):
            candidates.append(obj)
    h5.visititems(visit)
    if not candidates:
        raise RuntimeError("No numeric 3-D dataset found")
    preferred = [d for d in candidates if d.name.rstrip("/").split("/")[-1].lower() == "image"]
    return max(preferred or candidates, key=lambda d: int(np.prod(d.shape)))


def extract_canonical_crop(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    z0, x0, y0 = map(int, EXPECTED_ZXY0)
    zslice = slice(z0 - Z_HALF, z0 + Z_HALF + 1)
    xslice = slice(x0 - XY_HALF, x0 + XY_HALF + 1)
    yslice = slice(y0 - XY_HALF, y0 + XY_HALF + 1)
    with h5py.File(path, "r") as h5:
        ds = find_image_dataset(h5)
        shape = tuple(map(int, ds.shape))
        if shape == (801, 512, 512):
            crop = np.asarray(ds[zslice, xslice, yslice])
            storage = "canonical_zxy"
        elif shape == (512, 512, 801):
            # MATLAB v7.3 reverses dimensions: HDF5 [y,x,z] -> canonical [z,x,y].
            crop = np.asarray(ds[yslice, xslice, zslice]).transpose(2, 1, 0)
            storage = "matlab_v73_yxz"
        else:
            raise RuntimeError(f"Unexpected dataset shape {shape} at {ds.name}")
        return crop.astype(np.float64, copy=False), {
            "dataset": ds.name,
            "stored_shape": list(shape),
            "storage_convention": storage,
            "dtype": str(ds.dtype),
        }


def halfmax_width(y: np.ndarray, spacing: float) -> float:
    v = np.asarray(y, float)
    v = np.maximum(v - np.percentile(v, 5), 0)
    if not np.isfinite(v).all() or v.max() <= 0:
        return float("nan")
    v /= v.max()
    p = int(np.argmax(v)); left = p; right = p
    while left > 0 and v[left] >= 0.5:
        left -= 1
    while right < v.size - 1 and v[right] >= 0.5:
        right += 1
    if left == p or right == p:
        return float("nan")
    def crossing(a: int, b: int) -> float:
        return a + (0.5 - v[a]) / (v[b] - v[a]) if v[b] != v[a] else float(a)
    return float((crossing(right - 1, right) - crossing(left, left + 1)) * spacing)


def fit_gaussian3d(intensity: np.ndarray) -> dict[str, float | list[float]]:
    nz, nx, ny = intensity.shape
    z, x, y = np.indices(intensity.shape, dtype=float)
    pmax = np.unravel_index(np.argmax(intensity), intensity.shape)
    bg0 = float(np.percentile(intensity, 10)); amp0 = float(max(intensity.max() - bg0, 1e-12))
    p0 = np.asarray([amp0, *map(float, pmax), 2.5, 2.5, 2.5, bg0])
    lo = np.asarray([0, 0, 0, 0, 0.3, 0.3, 0.3, -np.inf])
    hi = np.asarray([np.inf, nz-1, nx-1, ny-1, nz, nx, ny, np.inf])
    def residual(p: np.ndarray) -> np.ndarray:
        amp, z0, x0, y0, sz, sx, sy, bg = p
        model = bg + amp * np.exp(-0.5 * (((z-z0)/sz)**2 + ((x-x0)/sx)**2 + ((y-y0)/sy)**2))
        return (model - intensity).ravel()
    res = least_squares(residual, p0, bounds=(lo, hi), loss="soft_l1", max_nfev=400)
    amp,z0,x0,y0,sz,sx,sy,bg = res.x
    factor = 2*np.sqrt(2*np.log(2))
    return {
        "success": bool(res.success), "cost": float(res.cost),
        "center_local_zxy": [float(z0),float(x0),float(y0)],
        "fwhm_z_um": float(factor*sz*DZ_UM),
        "fwhm_x_um": float(factor*sx*DX_UM),
        "fwhm_y_um": float(factor*sy*DX_UM),
        "amplitude": float(amp), "background": float(bg),
    }


def quantify(crop_amplitude: np.ndarray) -> tuple[dict[str, object], np.ndarray, tuple[int,int,int]]:
    # Deposited processed files are amplitude; the publication fits OCT intensity.
    intensity = np.maximum(crop_amplitude, 0)**2
    cz, cx, cy = np.asarray(intensity.shape)//2
    search = intensity[max(0,cz-8):cz+9,max(0,cx-8):cx+9,max(0,cy-8):cy+9]
    q = np.unravel_index(np.argmax(search), search.shape)
    peak = (cz-8+q[0],cx-8+q[1],cy-8+q[2])
    z,x,y = peak
    roi = intensity[max(0,z-10):z+11,max(0,x-10):x+11,max(0,y-10):y+11]
    direct = {
        "peak_local_zxy": list(map(int, peak)),
        "peak_global_saved_zxy0": list(map(int, EXPECTED_ZXY0 + np.asarray(peak)-np.asarray([Z_HALF,XY_HALF,XY_HALF]))),
        "fwhm_z_um": halfmax_width(intensity[:,x,y], DZ_UM),
        "fwhm_x_um": halfmax_width(intensity[z,:,y], DX_UM),
        "fwhm_y_um": halfmax_width(intensity[z,x,:], DX_UM),
        "peak_intensity": float(intensity[peak]),
        "gaussian_fit_21cube": fit_gaussian3d(roi),
    }
    return direct, intensity, peak


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cache = root / ".cache" / "processed_fig4"
    out = root / "reports" / "published_fig4_point"
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "experiment": "authors_processed_full_volume_Figure4_outfocus_point",
        "source_record": "10.5281/zenodo.7870795",
        "expected_global_saved_zxy0": EXPECTED_ZXY0.tolist(),
        "spacings_um": {"z": DZ_UM, "x": DX_UM, "y": DX_UM},
        "boundary": "Final amplitude outputs supplied by the authors; validates published output morphology, not raw-camera preprocessing or unmeasured spectral truth.",
        "methods": {},
    }
    plot_data = []
    for method, (name, md5) in FILES.items():
        path = download(name, md5, cache)
        crop, meta = extract_canonical_crop(path)
        metrics, intensity, peak = quantify(crop)
        report["methods"][method] = {"file": name, "md5": digest(path), "hdf5": meta, **metrics}
        plot_data.append((method, intensity, peak))
        np.savez_compressed(out / f"{method.lower()}_fig4_point_crop.npz", amplitude=crop, intensity=intensity,
                            expected_global_saved_zxy0=EXPECTED_ZXY0, spacing_z_um=DZ_UM, spacing_xy_um=DX_UM)
        path.unlink()
    # Identical, physically scaled line plots.
    plt.figure(figsize=(9.5,5.5))
    for method,intensity,peak in plot_data:
        z,x,y=peak; p=intensity[:,x,y]; p=p/max(float(p.max()),1e-30)
        coord=(np.arange(p.size)-z)*DZ_UM
        plt.plot(coord,p,label=method)
    plt.xlim(-15,15);plt.xlabel("z relative to local peak (µm)");plt.ylabel("normalized intensity")
    plt.title("Authors' deposited full-volume processed outputs: Figure-4 out-of-focus point")
    plt.legend();plt.tight_layout();plt.savefig(out/"published_fig4_axial_profiles.png",dpi=180);plt.close()
    (out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__ == "__main__":
    main()
