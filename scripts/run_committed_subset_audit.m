% Exact deposited FIAA/MIAA audits on the committed TiO2 subset.
% The subset is already phase corrected. This does not audit raw-camera preprocessing.
pkg load signal;

subset_file = getenv('SUBSET_MAT');
code_dir = getenv('UPSTREAM_CODE_DIR');
out_dir = getenv('OUTPUT_DIR');
if isempty(subset_file) || isempty(code_dir)
  error('SUBSET_MAT and UPSTREAM_CODE_DIR are required');
end
if isempty(out_dir)
  out_dir = 'reports/committed_subset_audit';
end
addpath(code_dir);
load(subset_file, 'x_strip', 'z_lines', 'sk', 'x_strip_y_zero_based', ...
  'x_strip_x_start_zero_based', 'coordinates_yx_zero_based');

Nz = 200;
q_i = 10;
q_rec = 2;
eta_weight = 1.0;
sk = sk(:);
b = sort(sk, 'descend');
mask = sk >= b(128);
ii = (find(mask, 1, 'first'):find(mask, 1, 'last'))';
if length(ii) ~= 128
  error('expected a contiguous 128-bin author source window');
end

% Central 48 adjacent A-lines include the strong point and both traversal directions.
sel = 41:88;
z_strip = x_strip(:, sel);
iRaw = flip(fft(z_strip, [], 1), 1);
C = complex(zeros(size(iRaw)));
sk0 = sk' * sk;
for i = 1:size(iRaw, 2)
  gk = (iRaw(:, i)' * sk) / sk0;
  C(:, i) = iRaw(:, i) - gk * sk;
end
anC = C ./ sk;
demod = anC(ii, :) - mean(anC(ii, :), 1);
line_scores = max(abs(C), [], 1);
[~, strong_idx] = max(line_scores);
x_indices_zero_based = (x_strip_x_start_zero_based + sel - 1)';

super = 4;
K = Nz * super;
K2 = 2 * K;
Nline = size(demod, 2);

maps_forward = recursive_maps_committed(demod, K2, q_i, q_rec, eta_weight);
maps_reverse = recursive_maps_committed(demod(:, end:-1:1), K2, q_i, q_rec, eta_weight);
maps_reverse = maps_reverse(:, end:-1:1);
maps_independent = complex(zeros(K2, Nline));
for i = 1:Nline
  [~, ~, amap] = fiaa_oct_c1(demod(:, i), K2, q_i, eta_weight);
  maps_independent(:, i) = amap;
  if mod(i, 10) == 0
    fprintf('independent line %d/%d complete\n', i, Nline);
  end
end

% Author's public wrapper path, including L=4 chunks, when Octave accepts it.
chunk_exact_ok = 0;
chunk_exact_error = '';
maps_chunk_forward = complex(nan(K2, Nline));
maps_chunk_reverse = complex(nan(K2, Nline));
try
  [~, maps_chunk_forward] = oct_iaa_c1(demod, q_i, K2, eta_weight, 2, q_rec, 4);
  [~, tmp_chunk_reverse] = oct_iaa_c1(demod(:, end:-1:1), q_i, K2, eta_weight, 2, q_rec, 4);
  maps_chunk_reverse = tmp_chunk_reverse(:, end:-1:1);
  chunk_exact_ok = 1;
catch err
  chunk_exact_error = err.message;
  fprintf('oct_iaa_c1 chunk path unavailable: %s\n', chunk_exact_error);
end

spectra_forward = maps_to_spectra_committed(maps_forward, K, Nz, ii(1), super);
spectra_reverse = maps_to_spectra_committed(maps_reverse, K, Nz, ii(1), super);
spectra_independent = maps_to_spectra_committed(maps_independent, K, Nz, ii(1), super);
if chunk_exact_ok
  spectra_chunk_forward = maps_to_spectra_committed(maps_chunk_forward, K, Nz, ii(1), super);
  spectra_chunk_reverse = maps_to_spectra_committed(maps_chunk_reverse, K, Nz, ii(1), super);
else
  spectra_chunk_forward = complex(nan(K, Nline));
  spectra_chunk_reverse = complex(nan(K, Nline));
end

save('-mat7-binary', [out_dir '/order.mat'], 'Nz', 'K', 'super', 'q_i', 'q_rec', ...
  'eta_weight', 'sk', 'ii', 'sel', 'x_indices_zero_based', 'x_strip_y_zero_based', ...
  'line_scores', 'strong_idx', 'C', 'anC', 'demod', 'chunk_exact_ok', ...
  'chunk_exact_error', 'spectra_forward', 'spectra_reverse', 'spectra_independent', ...
  'spectra_chunk_forward', 'spectra_chunk_reverse');

% Target-support audit on four checksum-traced selected A-lines.
iRaw4 = flip(fft(z_lines, [], 1), 1);
C4 = complex(zeros(size(iRaw4)));
anC4 = complex(zeros(size(iRaw4)));
for i = 1:4
  gk = (iRaw4(:, i)' * sk) / sk0;
  C4(:, i) = iRaw4(:, i) - gk * sk;
  anC4(:, i) = C4(:, i) ./ sk;
end
demod4 = anC4(ii, :) - mean(anC4(ii, :), 1);
supers = [2, 3, 4, 6, 8];
maxK = Nz * max(supers);
spectra_super = complex(nan(maxK, 4, length(supers)));
for si = 1:length(supers)
  su = supers(si);
  Ks = Nz * su;
  K2s = 2 * Ks;
  maps = complex(zeros(K2s, 4));
  for i = 1:4
    [~, ~, amap] = fiaa_oct_c1(demod4(:, i), K2s, q_i, eta_weight);
    maps(:, i) = amap;
  end
  spectra_super(1:Ks, :, si) = maps_to_spectra_committed(maps, Ks, Nz, ii(1), su);
  fprintf('super=%d complete\n', su);
end
save('-mat7-binary', [out_dir '/superfactor.mat'], 'Nz', 'supers', 'q_i', ...
  'eta_weight', 'sk', 'ii', 'C4', 'anC4', 'demod4', ...
  'coordinates_yx_zero_based', 'spectra_super');

fprintf('COMMITTED_SUBSET_AUDIT_OK output=%s\n', out_dir);
