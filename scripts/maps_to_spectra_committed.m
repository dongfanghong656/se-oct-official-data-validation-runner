function spectra = maps_to_spectra_committed(maps, K, Nz, ii_first, super)
  K2 = size(maps, 1);
  Nline = size(maps, 2);
  spectra = complex(zeros(K, Nline));
  shift_amount = round(Nz * ((super - 1) / 2)) + ii_first - 1;
  for i = 1:Nline
    fa = [maps(1, i); maps(end:-1:2, i)];
    xr = ifft(fa, [], 1) * K2;
    xr = circshift(xr, shift_amount, 1);
    spectra(:, i) = xr(1:K);
  end
end
