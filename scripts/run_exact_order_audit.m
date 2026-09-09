% Exact author FIAA/MIAA recursion-order audit on deterministic public-data strips.
% DATA_FILE: checksum-verified phase-corrected Cscan. DATASET_KIND: 'tio2' or 'leaf'.
% This does not audit raw-camera k-linearization, dispersion, or coverslip correction.
pkg load signal;

data_file = getenv('DATA_FILE');
code_dir = getenv('UPSTREAM_CODE_DIR');
out_file = getenv('OUTPUT_MAT');
kind = lower(getenv('DATASET_KIND'));
if isempty(data_file) || isempty(code_dir) || isempty(kind)
  error('DATA_FILE, UPSTREAM_CODE_DIR, and DATASET_KIND are required');
end
if isempty(out_file), out_file = ['reports/order_audit/' kind '_order_audit.mat']; end
addpath(code_dir);
load(data_file, 'Cscan', 'sk');
if ~isequal(size(Cscan), [200, 512, 512]), error('Unexpected Cscan shape'); end

Nz = 200; super = 4; K = Nz * super; K2 = 2 * K;
q_i = 10; q_rec = 2; eta_weight = 1.0;
iRawDatas = flip(fft(Cscan, [], 1), 1);
sk = sk(:);
if strcmp(kind, 'tio2')
  sk = flip(sk);
elseif ~strcmp(kind, 'leaf')
  error('DATASET_KIND must be tio2 or leaf');
end
b = sort(sk, 'descend'); mask = sk >= b(128);
ii = (find(mask, 1, 'first'):find(mask, 1, 'last'))';
if length(ii) ~= 128, error('Expected 128 contiguous source bins'); end

% Choose three deterministic B-scans from coarse low/mid/high signal-score strata.
amp = squeeze(max(abs(Cscan), [], 1)); % dimensions: x, y (MATLAB order)
y_candidates = (17:32:497)';
y_scores = zeros(length(y_candidates), 1);
for q = 1:length(y_candidates)
  yy = y_candidates(q);
  y_scores(q) = median(amp(:, yy));
end
[~, ord] = sort(y_scores);
sel_ord = [ord(2); ord(round(length(ord)/2)); ord(end-1)];
y_selected = y_candidates(sel_ord);
y_group = [1; 2; 3]; % low, mid, high coarse signal

strip_half = 24; Nstrip = 2 * strip_half + 1;
Nstrips = length(y_selected);
x_ranges = zeros(Nstrips, 2);
strip_scores = zeros(Nstrips, 1);
for s = 1:Nstrips
  yy = y_selected(s);
  v = amp(:, yy);
  kernel = ones(Nstrip, 1) / Nstrip;
  mov = conv(v, kernel, 'same');
  mov(1:strip_half) = -Inf; mov(end-strip_half+1:end) = -Inf;
  [strip_scores(s), xc] = max(mov);
  x_ranges(s, :) = [xc-strip_half, xc+strip_half];
end

% Outputs: K x Nstrip x Nstrips for each processing order.
spectra_forward = complex(zeros(K, Nstrip, Nstrips));
spectra_reverse = complex(zeros(K, Nstrip, Nstrips));
spectra_independent = complex(zeros(K, Nstrip, Nstrips));
normalized_full = complex(zeros(Nz, Nstrip, Nstrips));
raw_spectrum = complex(zeros(Nz, Nstrip, Nstrips));
line_scores = zeros(Nstrip, Nstrips);

for s = 1:Nstrips
  yy = y_selected(s);
  xs = x_ranges(s,1):x_ranges(s,2);
  raw = squeeze(iRawDatas(:, xs, yy));
  C = complex(zeros(size(raw)));
  sk0 = sk' * sk;
  for i = 1:Nstrip
    gk = (raw(:,i)' * sk) / sk0;
    C(:,i) = raw(:,i) - gk * sk;
  end
  anC = C ./ sk;
  demod = anC(ii,:) - mean(anC(ii,:), 1);
  normalized_full(:,:,s) = anC;
  raw_spectrum(:,:,s) = C;
  line_scores(:,s) = max(abs(C), [], 1)';

  % Forward recursive RFIAA/MIAA.
  maps = complex(zeros(K2, Nstrip));
  [a, PE, amap] = fiaa_oct_c1(demod(:,1), K2, q_i, eta_weight);
  maps(:,1) = amap; prev_power = abs(a).^2; prev_eta = PE(q_i+1);
  for i = 2:Nstrip
    [a, prev_eta, amap] = rec_fiaa_oct_c1(demod(:,i), K2, q_rec, eta_weight, prev_power, prev_eta);
    maps(:,i) = amap; prev_power = abs(a).^2;
  end
  spectra_forward(:,:,s) = maps_to_spectra(maps, K, Nz, ii(1));

  % Reverse recursive traversal, restored to original spatial order.
  demod_rev = demod(:, end:-1:1);
  maps_rev = complex(zeros(K2, Nstrip));
  [a, PE, amap] = fiaa_oct_c1(demod_rev(:,1), K2, q_i, eta_weight);
  maps_rev(:,1) = amap; prev_power = abs(a).^2; prev_eta = PE(q_i+1);
  for i = 2:Nstrip
    [a, prev_eta, amap] = rec_fiaa_oct_c1(demod_rev(:,i), K2, q_rec, eta_weight, prev_power, prev_eta);
    maps_rev(:,i) = amap; prev_power = abs(a).^2;
  end
  tmp = maps_to_spectra(maps_rev, K, Nz, ii(1));
  spectra_reverse(:,:,s) = tmp(:, end:-1:1);

  % Independent 10-iteration FIAA/MIAA per A-line: order-invariant reference.
  maps_ind = complex(zeros(K2, Nstrip));
  for i = 1:Nstrip
    [~, ~, amap] = fiaa_oct_c1(demod(:,i), K2, q_i, eta_weight);
    maps_ind(:,i) = amap;
  end
  spectra_independent(:,:,s) = maps_to_spectra(maps_ind, K, Nz, ii(1));
  fprintf('%s strip %d/%d y=%d x=%d:%d score=%g\n', kind, s, Nstrips, yy, xs(1), xs(end), strip_scores(s));
end

coords_meta = struct('y_selected', y_selected, 'y_group', y_group, 'x_ranges', x_ranges, ...
  'strip_scores', strip_scores, 'strip_half', strip_half);
save('-mat7-binary', out_file, 'kind', 'Nz', 'K', 'K2', 'q_i', 'q_rec', 'eta_weight', ...
  'sk', 'ii', 'coords_meta', 'line_scores', 'normalized_full', 'raw_spectrum', ...
  'spectra_forward', 'spectra_reverse', 'spectra_independent');
fprintf('ORDER_AUDIT_OK kind=%s strips=%d lines_per_strip=%d output=%s\n', kind, Nstrips, Nstrip, out_file);

function spectra = maps_to_spectra(maps, K, Nz, ii_first)
  K2 = size(maps,1); Nline = size(maps,2);
  spectra = complex(zeros(K, Nline));
  shift_amount = round(Nz * ((4-1)/2)) + ii_first - 1;
  for i = 1:Nline
    fa = [maps(1,i); maps(end:-1:2,i)];
    xr = ifft(fa, [], 1) * K2;
    xr = circshift(xr, shift_amount, 1);
    spectra(:,i) = xr(1:K);
  end
end
