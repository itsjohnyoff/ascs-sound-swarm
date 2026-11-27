# Soak 10 min, 4 virtual phones, speed 1x, seed 0

- server alive at end: True
- server memory MB: start 187, end 107, peak 208, growth -80.9 (limit 50), 61 samples
- dropped frames: {'p1': 0, 'p2': 0, 'p3': 0, 'p4': 0}
- calibrations: 19/19 ok; max residual per chirp ms: median 0.000, worst 0.012
- detections: 38/38 (100%), unmatched 0
- error m: mean 0.064, p90 0.061, max 0.624
- clap latency s, emission to dashboard: median 0.517, p90 0.541 (TARGET < 0.6)
- drone latency s, emission to dashboard: median 0.870, p90 0.893
- server health rtt: worst 0.054 s at +191 s; stalls > 1 s: none; client loop lag worst 2.870 s at +106 s
- phone streams: all finished

- MUST no crash: PASS
- MUST no phone disconnects: PASS
- MUST no memory growth: PASS
- MUST every chirp after the first calibrates: PASS
- MUST detection rate >= 90%: PASS
