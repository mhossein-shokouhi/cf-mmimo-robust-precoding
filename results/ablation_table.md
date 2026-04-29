## Ablation at $K=24$, $L=25$, $\tau_\mathrm{p}=4$ (10 seeds)

Gain column: point estimate is $(\bar{A}-\bar{B})/\bar{B}\times 100$ (table-consistent); 95% CI and $t$-statistic are computed from the paired difference $A_i - B_i$.

| Scheme | Precoder | Pilot Assignment | Mean throughput (bits/s/Hz) | Std | Δ vs *CF-WMMSE, MA-DRL PA* |
| --- | --- | --- | ---: | ---: | --- |
| Proposed Algorithm | Robust WMMSE | Proposed (heuristic) | 121.244 | 18.705 | +31.15% (CI ±3.43%, t=17.82) |
| Robust WMMSE, MA-DRL PA | Robust WMMSE | MA-DRL | 106.065 | 18.122 | +14.73% (CI ±0.82%, t=35.10) |
| CF-WMMSE, Proposed PA | CF-WMMSE | Proposed (heuristic) | 108.872 | 18.697 | +17.77% (CI ±3.64%, t= 9.55) |
| CF-WMMSE, MA-DRL PA | CF-WMMSE | MA-DRL |  92.447 | 18.587 | (baseline) |
