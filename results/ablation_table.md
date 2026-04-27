## Ablation at $K=24$, $L=25$, $\tau_\mathrm{p}=4$ (10 seeds)

Gain column: point estimate is $(\bar{A}-\bar{B})/\bar{B}\times 100$ (table-consistent); 95% CI and $t$-statistic are computed from the paired difference $A_i - B_i$.

| Scheme | Precoder | Pilot Assignment | Mean throughput (bits/s/Hz) | Std | Δ vs *CF-WMMSE, Random PA* |
| --- | --- | --- | ---: | ---: | --- |
| Proposed Algorithm | Robust WMMSE | Proposed | 121.376 | 17.515 | +35.69% (CI ±4.67%, t=14.97) |
| Robust WMMSE, Random PA | Robust WMMSE | Random | 104.477 | 19.365 | +16.80% (CI ±1.50%, t=21.97) |
| CF-WMMSE, Proposed PA | CF-WMMSE | Proposed | 108.593 | 16.762 | +21.40% (CI ±5.46%, t= 7.68) |
| CF-WMMSE, Random PA | CF-WMMSE | Random | 89.451 | 18.753 | (baseline) |
