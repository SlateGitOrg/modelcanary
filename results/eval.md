## Evaluation: 1000 held-out seeds x 7 scenarios x 3 strategies (21000 simulated rollouts)

Pilot (A/A, tuning seeds 0..19) planning sd: `error_rate` = 0.4255, `latency_ms` = 44.5916, `conversion` = 0.4603

Evaluation seeds start at 1000000; wall time 325.9 s.

### Error rates by strategy

| strategy | false-promotion rate [95% CI] | false-rollback rate [95% CI] | inconclusive | mean requests (good) | mean requests (worse) | mean candidate exposure (worse) |
|---|---|---|---|---|---|---|
| sequential | 0.001 [0.000, 0.002] | 0.011 [0.008, 0.016] | 0.000 | 39,396 | 7,562 | 2,125 |
| fixed_horizon | 0.000 [0.000, 0.001] | 0.020 [0.016, 0.026] | 0.000 | 29,088 | 16,344 | 4,543 |
| peeking | 0.006 [0.004, 0.009] | 0.170 [0.157, 0.184] | 0.000 | 10,829 | 2,830 | 294 |

### Per scenario

| scenario | truth | strategy | runs | P(promote) | mean req | median req | p90 req | mean candidate req |
|---|---|---|---|---|---|---|---|---|
| better | better | fixed_horizon | 1000 | 1.000 | 29,152 | 29,000 | 29,500 | 9,778 |
| better | better | peeking | 1000 | 0.977 | 10,054 | 9,000 | 14,500 | 1,274 |
| better | better | sequential | 1000 | 0.998 | 23,146 | 21,000 | 37,000 | 6,928 |
| better-error-only | better | fixed_horizon | 1000 | 0.975 | 29,090 | 29,000 | 29,500 | 9,747 |
| better-error-only | better | peeking | 1000 | 0.796 | 11,432 | 10,500 | 18,000 | 1,868 |
| better-error-only | better | sequential | 1000 | 0.996 | 47,804 | 42,000 | 84,050 | 19,155 |
| broken | worse | fixed_horizon | 1000 | 0.000 | 501 | 500 | 500 | 0 |
| broken | worse | peeking | 1000 | 0.000 | 501 | 500 | 500 | 0 |
| broken | worse | sequential | 1000 | 0.000 | 501 | 500 | 500 | 0 |
| equal | equal | fixed_horizon | 1000 | 0.965 | 29,022 | 29,000 | 29,500 | 9,713 |
| equal | equal | peeking | 1000 | 0.718 | 11,000 | 10,500 | 17,500 | 1,787 |
| equal | equal | sequential | 1000 | 0.972 | 47,237 | 40,750 | 83,000 | 18,865 |
| worse-conversion | worse | fixed_horizon | 1000 | 0.001 | 29,022 | 29,000 | 29,500 | 9,713 |
| worse-conversion | worse | peeking | 1000 | 0.023 | 9,160 | 8,500 | 15,000 | 1,162 |
| worse-conversion | worse | sequential | 1000 | 0.002 | 25,969 | 24,000 | 42,500 | 8,351 |
| worse-error | worse | fixed_horizon | 1000 | 0.000 | 22,383 | 22,500 | 22,500 | 6,392 |
| worse-error | worse | peeking | 1000 | 0.000 | 1,074 | 1,000 | 1,500 | 15 |
| worse-error | worse | sequential | 1000 | 0.000 | 2,814 | 2,000 | 7,000 | 144 |
| worse-latency | worse | fixed_horizon | 1000 | 0.000 | 13,468 | 13,500 | 13,500 | 2,067 |
| worse-latency | worse | peeking | 1000 | 0.000 | 586 | 500 | 1,000 | 0 |
| worse-latency | worse | sequential | 1000 | 0.000 | 966 | 1,000 | 1,500 | 4 |
