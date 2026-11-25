# Benchmark with the heuristic classifier live (2026-11-25)

Re-run of `python -m sim.benchmark` after the classifier landed; the 2026-11-24 report predates it (every class was `other`).

- Seeds 0-4 (clean) and seed 5 with `--occlude p2`: 60/60 sounds classified correctly (30 claps, 30 drones), 0/60 missed.
- Localization: clean seeds mean 0.001-0.002 m; occluded seed 5 mean 0.095 m, p90 0.205 m.
- Classifier alone (`SoundClassifier.classify_swarm`, 4 phones, 50 runs each): clap median 1.47 ms (max 2.51), drone median 8.21 ms (max 10.81). MUST < 20 ms per event: PASS. MUST > 90% clap vs drone on simulator: PASS (100%).

## seed 0

| # | gt | est | error_m | expected | kind | class | inside_hull |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | (3.00, 2.50) | (3.00, 2.50) | 0.002 | clap | transient | clap | True |
| 2 | (5.50, 3.50) | (5.50, 3.50) | 0.001 | drone | sustained | drone | True |
| 3 | (2.00, 4.00) | (2.00, 4.00) | 0.002 | clap | transient | clap | True |
| 4 | (4.50, 1.80) | (4.50, 1.80) | 0.002 | drone | sustained | drone | True |
| 5 | (6.00, 2.20) | (6.00, 2.20) | 0.001 | clap | transient | clap | True |
| 6 | (3.50, 4.20) | (3.50, 4.20) | 0.002 | drone | sustained | drone | True |
| 7 | (4.00, 3.00) | (4.00, 3.00) | 0.001 | clap | transient | clap | True |
| 8 | (2.20, 2.00) | (2.20, 2.00) | 0.002 | drone | sustained | drone | True |
| 9 | (0.40, 2.80) | (0.40, 2.80) | 0.003 | clap | transient | clap | False |
| 10 | (7.60, 3.40) | (7.60, 3.40) | 0.003 | drone | sustained | drone | False |

mean 0.002 m, median 0.002 m, p90 0.003 m, missed 0/10, offline runtime 450 ms/event incl. full-recording onset scan

## seed 5, occlude p2

| # | gt | est | error_m | expected | kind | class | inside_hull |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | (3.00, 2.50) | (3.01, 2.48) | 0.025 | clap | transient | clap | True |
| 2 | (5.50, 3.50) | (5.52, 3.46) | 0.046 | drone | sustained | drone | True |
| 3 | (2.00, 4.00) | (1.98, 4.01) | 0.025 | clap | transient | clap | True |
| 4 | (4.50, 1.80) | (4.57, 1.70) | 0.122 | drone | sustained | drone | True |
| 5 | (6.00, 2.20) | (5.88, 2.35) | 0.197 | clap | transient | clap | True |
| 6 | (3.50, 4.20) | (3.50, 4.19) | 0.009 | drone | sustained | drone | True |
| 7 | (4.00, 3.00) | (3.96, 3.05) | 0.061 | clap | transient | clap | True |
| 8 | (2.20, 2.00) | (2.17, 2.02) | 0.036 | drone | sustained | drone | True |
| 9 | (0.40, 2.80) | (0.25, 2.82) | 0.149 | clap | transient | clap | False |
| 10 | (7.60, 3.40) | (7.35, 3.52) | 0.278 | drone | sustained | drone | False |

mean 0.095 m, median 0.054 m, p90 0.205 m, missed 0/10, offline runtime 444 ms/event incl. full-recording onset scan
