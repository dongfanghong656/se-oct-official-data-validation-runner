function spectra = maps_to_spectra_committed(maps, K, Nz, ii_first, super, wrapper_reversed)
  % Convert MAP output to the author MIAA spectrum.
  %
  % fiaa_oct_c1 / rec_fiaa_oct_c1 return the raw MAP ordering. oct_iaa_c1
  % reverses rows [1,end:-1:2] before returning. The publication MIAA path
  % reverses the wrapper output once more before IFFT. Therefore:
  %   raw direct maps      -> IFFT directly;
  %   oct_iaa_c1 outputs   -> undo wrapper reversal once, then IFFT.
  if nargin < 6
    wrapper_reversed = false;
  end
  K2 = size(maps, 1);
  Nline = size(maps, 2);
  spectra = complex(zeros(K, Nline));
  shift_amount = round(Nz * ((super - 1) / 2)) + ii_first - 1;
  for i = 1:Nline
    if wrapper_reversed
      fa = [maps(1, i); maps(end:-1:2, i)];
    else
      fa = maps(:, i);
    end
    xr = ifft(fa, [], 1) * K2;
    xr = circshift(xr, shift_amount, 1);
    spectra(:, i) = xr(1:K);
  end
end
