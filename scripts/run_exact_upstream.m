% Exact selected-A-line execution of the authors' deposited FIAA/MIAA routines.
% Public Zenodo data only. The input volume is already phase corrected.

pkg load signal;

data_file = getenv('DATA_FILE');
code_dir = getenv('UPSTREAM_CODE_DIR');
out_file = getenv('OUTPUT_MAT');
if isempty(data_file) || isempty(code_dir)
  error('DATA_FILE and UPSTREAM_CODE_DIR are required');
end
if isempty(out_file)
  out_file = 'reports/exact/author_exact_alines.mat';
end
addpath(code_dir);

load(data_file, 'Cscan', 'sk');
if ~isequal(size(Cscan), [200, 512, 512])
  error('Unexpected Cscan shape');
end

coords_yx0 = [12,7; 232,210; 482,38; 1,453];
coords_yx1 = coords_yx0 + 1;
Nz = size(Cscan,1);
Nlines = size(coords_yx1,1);
super = 4;
K = Nz * super;
q_i = 10;
eta_weight = 1.0;

% Exact preprocessing in MIAA_ISAM_processing.m for the TiO2 data.
iRawDatas = flip(fft(Cscan, [], 1), 1);
sk = flip(sk(:));
b = sort(sk, 'descend');
threshold = b(128);
mask = sk >= threshold;
ii_first = find(mask, 1, 'first');
ii_last = find(mask, 1, 'last');
ii = (ii_first:ii_last)';
if length(ii) ~= 128
  error('Author source selection did not produce 128 contiguous bins');
end

% Exact axial edge taper used before ISAM.
hanning_edge = hanning(200);
edge_apod = ones(K,1);
edge_apod(1:100) = hanning_edge(1:100);
edge_apod(K-99:K) = hanning_edge(101:200);

raw_spectrum = complex(zeros(Nz,Nlines));
normalized_full = complex(zeros(Nz,Nlines));
demod_spectrum = complex(zeros(length(ii),Nlines));
dft_profile = complex(zeros(K,Nlines));
rfiaa_profile = complex(zeros(K,Nlines));
rfiaa_map_profile = complex(zeros(K,Nlines));
miaa_spectrum = complex(zeros(K,Nlines));
miaa_profile = complex(zeros(K,Nlines));
dispersed_miaa_profile = complex(zeros(K,Nlines));
corrected_miaa_profile = complex(zeros(K,Nlines));
pe_first = zeros(q_i+1,Nlines);
pe_second = zeros(q_i+1,Nlines);

beta = 24.0;
xi = linspace(-1,1,length(ii))';
phase_term = exp(1i * beta * xi.^2);

for n = 1:Nlines
  y = coords_yx1(n,1);
  x = coords_yx1(n,2);
  raw = iRawDatas(:,y,x);
  gk = (raw' * sk) / (sk' * sk);
  C = raw - gk * sk;
  raw_spectrum(:,n) = C;
  dft_profile(:,n) = ifft(detrend(C), K, 1);
  anC = C ./ sk;
  normalized_full(:,n) = anC;
  demod = anC(ii) - mean(anC(ii));
  demod_spectrum(:,n) = demod;

  [rf, pe1, rfmap] = fiaa_oct_c1(demod, K, q_i, eta_weight);
  rfiaa_profile(:,n) = rf;
  rfiaa_map_profile(:,n) = rfmap;
  pe_first(:,n) = pe1(:);

  [~, pe2, map2] = fiaa_oct_c1(demod, 2*K, q_i, eta_weight);
  pe_second(:,n) = pe2(:);
  fa_map = [map2(1); map2(end:-1:2)];
  xr2 = ifft(fa_map, [], 1) * (2*K);
  shift_amount = round(Nz * ((super-1)/2)) + ii_first - 1;
  xr2a = circshift(xr2, shift_amount, 1);
  xr2a = xr2a(1:K);
  miaa_spectrum(:,n) = xr2a;
  miaa_profile(:,n) = ifft(xr2a .* edge_apod, K, 1);

  demod_disp = demod .* phase_term;
  [~, ~, map_disp] = fiaa_oct_c1(demod_disp, 2*K, q_i, eta_weight);
  fa_disp = [map_disp(1); map_disp(end:-1:2)];
  xr_disp = ifft(fa_disp, [], 1) * (2*K);
  xr_disp = circshift(xr_disp, shift_amount, 1);
  xr_disp = xr_disp(1:K);
  dispersed_miaa_profile(:,n) = ifft(xr_disp .* edge_apod, K, 1);

  demod_corr = demod_disp .* conj(phase_term);
  [~, ~, map_corr] = fiaa_oct_c1(demod_corr, 2*K, q_i, eta_weight);
  fa_corr = [map_corr(1); map_corr(end:-1:2)];
  xr_corr = ifft(fa_corr, [], 1) * (2*K);
  xr_corr = circshift(xr_corr, shift_amount, 1);
  xr_corr = xr_corr(1:K);
  corrected_miaa_profile(:,n) = ifft(xr_corr .* edge_apod, K, 1);
end

save('-mat7-binary', out_file, 'coords_yx0', 'coords_yx1', 'Nz', 'K', 'super', ...
  'q_i', 'eta_weight', 'sk', 'threshold', 'mask', 'ii', 'ii_first', 'ii_last', ...
  'edge_apod', 'raw_spectrum', 'normalized_full', 'demod_spectrum', 'dft_profile', ...
  'rfiaa_profile', 'rfiaa_map_profile', 'miaa_spectrum', 'miaa_profile', ...
  'dispersed_miaa_profile', 'corrected_miaa_profile', 'beta', 'pe_first', 'pe_second');

fprintf('EXACT_UPSTREAM_OK lines=%d ii=%d:%d K=%d output=%s\n', Nlines, ii_first, ii_last, K, out_file);
