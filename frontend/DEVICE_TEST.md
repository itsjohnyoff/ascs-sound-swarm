# Real-device test checklist

Serve over HTTPS (`python scripts/start_https.py --mode lan`), open `https://<laptop-ip>:8443/phone.html?debug=1` on each phone.
The `?debug=1` panel shows sample rate, AEC/NS/AGC, frames, index gaps, reconnects and clock jumps. Mark each line `PASS`, `FAIL` or `N/A` and add a note.

Tester: ______  Date: ______  Server commit: ______

| # | Check | iOS Safari | Android Chrome |
| --- | --- | --- | --- |
| 1 | Page loads over HTTPS with no certificate or "not secure" warning blocking the mic | | |
| 2 | Tap Join: mic permission prompt appears and is granted | | |
| 3 | Debug panel: sampleRate noted here: ______ Hz (tell Agents 1 and 2 right away if 44100) | | |
| 4 | Debug panel: AEC off, NS off, AGC off (no "browser processing ON" badge). If any is ON, note which | | |
| 5 | No "mic constraints downgraded" badge | | |
| 6 | Phone id shown large and stays the same after a page reload | | |
| 7 | Phone appears on the dashboard map, VU meter moves when tapping near the mic | | |
| 8 | Index gaps stays 0 for 2 minutes of normal use | | |
| 9 | Dashboard Calibrate: sync pill turns green ("Synced x.xx ms") | | |
| 10 | Screen wake lock: screen stays on for 10 minutes untouched | | |
| 11 | Lock the screen for 10 s and unlock: stream resumes, clock jumps +1, pill goes "Needs re-sync" until the next chirp, then green | | |
| 12 | Switch to another app for 10 s and back: same behaviour as line 11 | | |
| 13 | Toggle Wi-Fi off 5 s then on: reconnects (reconnects +1), same phone id, same server position, re-sync after next chirp | | |
| 14 | Position shown on the phone page matches what the dashboard has | | |
| 15 | Clap near the phone with 4 phones joined: one pin appears at the right place | | |

## Projector check (judge mode)
| Check | Result |
| --- | --- |
| Judge mode readable from the back of the room (map, error line, cards) | |
| Nothing but the map and judge console visible (no debug panel without ?debug=1) | |
| Class badges (DRONE/CLAP/OTHER) distinguishable on the projector | |

## Failures
Describe each FAIL (device, OS/browser version, what the debug panel said):
