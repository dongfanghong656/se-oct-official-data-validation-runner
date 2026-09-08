# Exact strip rerun status

The first strip run reached the exact recursive FIAA call with an omitted `eta_in` argument and failed. Commit `98e3914737b3022ac5dc435b07877a4a2f0e32c4` corrected the call by carrying both prior-line power and prior noise state exactly as `rfiaa_oct_c1.m` does. A fresh public-data rerun was triggered at 2026-09-08T15:19:37Z. Results are pending; the stale failure log must not be treated as the fixed run result.
