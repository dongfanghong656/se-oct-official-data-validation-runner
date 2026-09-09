function maps = recursive_maps_committed(demod, K2, q_i, q_rec, eta_weight)
  Nline = size(demod, 2);
  maps = complex(zeros(K2, Nline));
  [a, PE, amap] = fiaa_oct_c1(demod(:, 1), K2, q_i, eta_weight);
  maps(:, 1) = amap;
  previous_power = abs(a).^2;
  previous_eta = PE(q_i + 1);
  for i = 2:Nline
    [a, previous_eta, amap] = rec_fiaa_oct_c1( ...
      demod(:, i), K2, q_rec, eta_weight, previous_power, previous_eta);
    maps(:, i) = amap;
    previous_power = abs(a).^2;
  end
end
