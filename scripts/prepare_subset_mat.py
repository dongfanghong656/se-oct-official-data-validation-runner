#!/usr/bin/env python3
"""Create a compact MAT input from the committed, checksum-traced TiO2 subset."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os

import numpy as np
from scipy.io import savemat


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_npz = root / "reports" / "extracts" / "official_tio2_selected_subset.npz"
    sk_path = root / "reports" / "extracts" / "author_sk_tio2.csv"
    meta_path = root / "reports" / "extracts" / "author_sk_tio2_meta.json"
    out_path = Path(os.environ.get("SUBSET_MAT", root / "reports" / "committed_subset_audit" / "subset.mat"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    actual_sk_sha = hashlib.sha256(sk_path.read_bytes()).hexdigest()
    if actual_sk_sha != meta["text_sha256"]:
        raise RuntimeError(f"author sk checksum mismatch: {actual_sk_sha}")
    with np.load(source_npz) as data:
        x_strip = np.asarray(data["x_strip"], dtype=np.complex64)
        z_lines = np.asarray(data["z_lines"], dtype=np.complex64)
        x_strip_y = int(np.asarray(data["x_strip_y"]).item())
        x_strip_x_start = int(np.asarray(data["x_strip_x_start"]).item())
        coords = np.asarray(data["coordinates_yx"], dtype=np.int32)
    sk = np.loadtxt(sk_path, dtype=np.float64)
    if x_strip.shape != (200, 129) or z_lines.shape != (200, 4) or sk.shape != (200,):
        raise RuntimeError(f"unexpected shapes: x_strip={x_strip.shape}, z_lines={z_lines.shape}, sk={sk.shape}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        out_path,
        {
            "x_strip": x_strip,
            "z_lines": z_lines,
            "sk": sk,
            "x_strip_y_zero_based": np.int32(x_strip_y),
            "x_strip_x_start_zero_based": np.int32(x_strip_x_start),
            "coordinates_yx_zero_based": coords,
        },
        do_compression=True,
    )
    print(json.dumps({
        "output": str(out_path),
        "x_strip_shape": list(x_strip.shape),
        "z_lines_shape": list(z_lines.shape),
        "author_sk_text_sha256": actual_sk_sha,
        "boundary": "phase-corrected public subset; not raw camera data",
    }, indent=2))


if __name__ == "__main__":
    main()
