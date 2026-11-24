# ASCS Sound Swarm

**ASCS Hackathon · 48-hour challenge**

Turn audience phones into a distributed microphone array. Participants open an HTTPS web page and grant microphone access. The app estimates where a clap or drone buzz came from and marks the estimated position on a live 2D floor map.

## What we need to build

1. Let phones join a session, mark their fixed positions on the floor map, and stream timestamped audio samples.
2. Use a calibration chirp to align audio across phones. Account for the chirp's travel time from its known position and check for clock drift during the demo.
3. Use GCC-PHAT to estimate differences in sound arrival time between phone pairs.
4. Use 2D multilateration to estimate the source position and show it on the map.
5. Add a classifier that labels a detected sound as drone-like or other.

## Tools and equipment

| Tool or equipment | What it is for |
| --- | --- |
| Web Audio API and AudioWorklet | Capture audio samples on each phone |
| WebSocket connection over HTTPS | Stream audio to the laptop and return location estimates |
| [ggwave](https://github.com/ggerganov/ggwave) | Acoustic signaling library to evaluate for calibration |
| GCC-PHAT | Estimate relative arrival delays from pairs of audio streams |
| 2D multilateration | Convert relative delays and known phone positions into a point on the map |
| DroneAudioSet and DREGON | Drone-audio datasets named in the challenge brief for classifier experiments |
| Four phones, a laptop, a speaker, Wi-Fi and a measured floor map | Record sound from fixed positions, play the reference chirp and run the demo |

## 48-hour build order

- **First milestone** — Capture a clap on two phones and verify the relative delay with a known source position.
- **Second milestone** — Place four phones around the room and get a clap to appear on the live map.
- **Final milestone** — Repeat calibration, add the drone label, and rehearse the judged run.

## Judging and main risk

Judges will measure 2D localization error at ten positions they choose. Record each ground-truth point, estimated point and distance error in meters. Track missed detections alongside those errors.

The brief calls out sub-5 ms synchronization as the main challenge. Test mixed phones early and measure residual timing error after calibration. Check how browser latency, clock drift and indoor echoes affect the estimate.
