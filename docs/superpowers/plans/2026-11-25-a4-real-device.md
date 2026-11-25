# Agent 4 plan: real-device diagnostics (2026-11-25)

## Contract map (frontend vs server)
| Message / endpoint | Frontend | Server | Status |
| --- | --- | --- | --- |
| Binary frame, 32 B header | `shared/binaryFrame.ts` | `shared/protocol.py` | Match, byte-tested against Python `struct.pack` |
| `phone_hello` | phone page, re-sends server-known position | `PhoneHello` | Match |
| `phone_status` | sync pill | `session.phone_status` | Match (no position field; page reads `/api/session`) |
| `session_state`, `telemetry` | dashboard | `session.snapshot/telemetry` | Match |
| `calibration_result` | table | `CalibrationResult` | Match |
| `detection` + `update_seq`, `kind` | dashboard | `events.py` | Match; sustained updates replace by `event_id` |
| `/api/eval/known-point` | known-point badge | exists | Match |
| `/api/detections/recent` | **was unused**: judge list empty after reload | exists | Fixed: loaded on boot |

## Deliverables
1. `?debug=1` phone diagnostics panel (`src/phone/diagnostics.ts`, Vitest).
2. `frontend/DEVICE_TEST.md` pass/fail checklist for iOS Safari and Android Chrome.
3. `?debug=1` draw-rate/cost meter on the dashboard (`src/dashboard/fps.ts`, Vitest), measured with 8 virtual phones.
4. Fix: restore recent detections after a dashboard reload.

## Gate
`npm test`, `npm run build` (zero TS errors), dist has index.html, phone.html, pcm-worklet.js; full stack with virtual phones shows a pin.
