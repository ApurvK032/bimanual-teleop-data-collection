# Troubleshooting

Start with the app's **System** tab (or `python -m bimanual_collect doctor`): it checks the PC and offers a fix for
each problem.

| Symptom | Likely cause | Fix |
|---|---|---|
| `the ZED is on USB 480 Mbps; it needs USB 3` | the USB-C plug at the camera isn't making its USB 3 contacts | flip and reseat the plug; use a USB 3 port; check `/sys/bus/usb/devices/*/speed` shows 5000 |
| `no ZED found` | not plugged in, or no permission on `/dev/video*` | plug in; System tab → **Install (password)** (camera access) and replug |
| `N ZEDs connected ... set camera serials` | two ZEDs on one PC | put each camera's `serial` in `configs/rig.yaml` (`cameras:` list) |
| `calibration file not found` | the factory calibration wasn't downloaded | System tab → **Download it** (or `python -m bimanual_collect calibration`) |
| `port 8443 (or 8444) is already in use` | the teleop repo's own `restart_localhost.sh` (or another program) is running | stop it; the app adopts its own services and restarts crashed ones |
| App: "No camera" | the ZED isn't plugged in (or not on USB 3) | plug it in: the app retries by itself and starts the recorder as soon as the camera appears |
| Arm control: "Arms answer on the network" FAIL | arms off, cable out, or the PC has no address on the arm network | switch the arms on; Bench step 1 (PC and network) shows the fix |
| Arm control: "PC ready" FAIL | not on mains power, power mode not *performance*, or the route to the arms goes over Wi-Fi | Bench step 1: **Set performance mode** / **Assign to arm network** |
| Arm control stays on "Waiting for the headset" | the page isn't streaming the right mode, a controller isn't tracked, or a grip is held | do what the card says (open the page, input *Bimanual*, Enter VR, release the grips); it gives up after 2 minutes |
| "The headset is not on USB" when starting | the arm program wakes the headset's tracking over USB | connect the Quest by USB-C and accept USB debugging |
| "Safety stop: ..." | the teleop program stopped itself (tracking error, collision hold, stale controller stream...) | read the reason; the arms returned to rest. Repeated stops on one joint: check for an obstacle or a loose cable before raising any limit |
| Red banner "Cut the arm power now" | the launcher could not return an arm to rest | cut the power at the switch with the arm supported; restart it later; don't retry from software |
| "Not responding after Stop" | the launcher is stuck in its shutdown | if the arms are not at rest, cut the power at the switch. The app never kills it after the arms have moved |
| "another program is connected to the arms" | a launcher or preflight started from a terminal is still running | stop it in its terminal (Ctrl+C): the arm controller takes one client at a time |
| "Install.sh" opens in a text editor | double-click opens scripts as text | right-click → **Run as a Program**, or run `./Install.sh` in a terminal |
| Console: camera pill red, "not delivering" | camera unplugged or USB error | plug it back in: the recorder reopens it every 2 s by itself. A running episode was closed as *aborted* |
| Many camera drops (QC: "camera drops > 2%") | USB bandwidth or CPU contention | use 1080 at 15 fps or 720 at 30 fps (1080 at 30 fps dropped 0.6-3.9% here); don't run processing while recording; keep the camera on its own USB controller |
| Warning "recording, but no arm run is active" | the episode started before the arm launcher | start the arm launcher; frames outside the arm run are trimmed at alignment |
| Warning "an arm run is active but nothing is being recorded" | A wasn't pressed | press **A** (both grips released) or **Start episode** in the app |
| A does nothing in the headset | a grip is held (grips are the deadman) | release both grips, then press A |
| QC "alignment: no teleop telemetry overlaps" | the episode had no arm run, or the PC rebooted between them | record with the arm launcher running; check `paths.teleop_runs` in the rig config |
| QC "sync ... r < 0.6" | little arm motion in the sync box, or the operator inside it | redraw the timing box around the arms only (Bench tab, Camera step) |
| Episode status *recovered* | the recorder process died mid-episode | nothing to do: it was repaired on the next start (about a second lost); review it |
| Export fails "episodes differ in cameras/modes" | episodes recorded in different camera modes or with different cameras | export them as separate datasets |
| Export job: "LeRobot environment not found" | `.venv-lerobot` missing | System tab → **Run the installer again**, or set `paths.lerobot_python` |
| Depth wildly wrong (e.g. table at 2.6 m) | the calibration file belongs to another camera, or a mode it doesn't cover | check the serial in `calibration/SN<serial>.conf` matches the camera (System tab), download it again |
| Quest page: "Recorder offline" | the camera service isn't running or the port forward to the headset is missing | start the app; Headset card → **Reconnect ports** |
