# Deposited leaf-output boundary

The checksum-verified Zenodo file `exp_leaf_image_MIAA_ISAM.mat` (MD5 `239a1e714b77e6aa6640f4dd2f50a644`) is a MATLAB v5/v7 compressed file whose `image` variable has shape `401 × 512 × 512`.

The deposited `plot_figure5.m` uses full-volume depth indices `Zindex1 = 720` and `Zindex2 = 637`. Therefore this 401-plane file is not, by itself, the complete depth volume consumed by the published Figure-5 plotting path. It appears to be a cropped downstream product (also relevant to CAO), and the exact published Figure-5 A-line cannot be independently re-extracted from this file alone.

Evidence: `runner.log` records the verified file-format/shape inspection; `reports/source_ranges/figure_parameters.md` records the plotting indices from the checksum-verified source. This is a reproducibility limitation, not an allegation that the paper's internal plot was fabricated.
