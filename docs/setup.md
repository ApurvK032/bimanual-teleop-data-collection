# Setup: from an empty PC to your first dataset

About an hour the first time, no terminal needed. You need the arm(s), a Quest 3, a ZED camera, and a Linux PC
(Ubuntu 24.04) with a free wired network port (a USB-Ethernet adapter is fine).

## 1. Install (15 minutes, once per PC)

1. Download this repository (green **Code** button → **Download ZIP**) and unpack it, e.g. in your home folder.
2. Right-click **Install.sh** → **Run as a Program**. A progress window builds everything inside that folder:
   the app, the arm driver environment and the dataset export environment.
3. If the PC still needs system parts (camera access rule, FFmpeg, adb), the installer offers to add them: Ubuntu
   asks for your password once.
4. At the end the app is in the app menu and on the desktop as **Robot data collection**.

(From a terminal instead: `git clone --recurse-submodules <repo>`, then `./Install.sh`.) A ZIP download has no
submodules; the installer fetches the missing pinned parts (such as Trossen's arm model) itself. For learned (RAFT) depth on an NVIDIA GPU, also run
`scripts/install.sh --gpu-depth`.

To update later: download or `git pull` the new version and run **Install.sh** again; your bench settings and data
stay where they are.

## 2. Connect the hardware

- **Camera:** ZED overhead, looking down at the table, in a **USB 3** port (blue or marked SS).
- **Arms:** both arm controllers on one network switch, the switch on the PC's free wired port. The arms must be
  on 192.168.1.x (Trossen's network) and each needs its own address; the PC takes 192.168.1.10 on that port.
  Mains power, not battery.
- **Headset:** Quest 3 by USB-C. Once per headset: turn on developer mode in the Meta Horizon phone app, then put
  the headset on and accept *Allow USB debugging* (tick *Always allow from this computer*).

## 3. Set up the bench (10 minutes, once per bench)

Click **Robot data collection**. The browser opens on the **Bench** tab; work down the steps, each checks itself:

1. **PC and network**: pick the wired port the arms are on (**Assign to arm network**). Set performance mode if asked.
2. **Arms on the network**: two arms or one; **Scan network** (ping only, nothing is sent to an arm).
3. **Which arm is which**: the cable test. Standing behind the arms, unplug the network cable of the arm on your
   left; the app sees which address went silent. Plug it back in. One arm: pick the controller hand.
4. **Layout**: measure the bases centre to centre with a tape and save. **Render simulated bench** shows it in 3D:
   the left arm in the picture must be the left arm on the table.
5. **Driving settings**: start with **Gentle (0.40)** on a new bench.
6. **Checks**: the teleop program's own setting checks, then the **read-only arm check** (connects, reads joints and
   versions, disconnects; nothing moves).
7. **First-run check** (two arms): each arm alone at 0.40 while the other stays read-only. Tick each motion that went
   the right way, then **Record validation**.
8. **Camera**: draw the timing box around the arms (keep yourself out of it), the video crop and the table. Redraw
   them whenever the camera moves.

## 4. Collect (the daily routine)

1. Click **Robot data collection**. The **Collect** tab shows the camera, the arms, the recording and the headset.
2. **Headset** panel: **Open page in headset**. In the headset, check the input is *Bimanual* (or your hand for one
   arm), then press **Enter Passthrough/VR**.
3. **Arm control** panel, **Live**: when every check shows OK, **Start arms**. Confirm the two points (arms at rest and
   nothing in reach; someone at the power switch). The arms start once both controllers are tracked with the grips
   released: they move to home, then each grip drives its arm.
4. Pick the task and your name. In the headset: **A** (both grips released) starts an episode, **A** stops it, then
   **A** = success, **B** = failure. The headset header shows the arm state and the recording timer.
5. **Stop arms** (on every tab, and in the top bar) returns the arms to rest. Each run also stops by itself after its
   time limit. Releasing a grip holds that arm where it is. The power switch is always the emergency stop.

**Stand behind the arms when you press Enter Passthrough.** The headset page takes the direction you face at that
moment as "forward" for the whole session. Entering while standing beside the arms (for example facing them at 90°)
turns every move: pushing your hand forward moves the arm sideways. If moves feel turned, release both grips, Exit VR,
stand behind the arms and Enter Passthrough again.

The app's **Controls** tab has a one-page operator card (every button, the order of a session, safety) with
Print / View PDF / Download PDF; print it for the bench.

In **Bimanual** mode, release both grips and hold **X** on the left controller to return both arms to their startup
home pose. Keep holding until the headset shows **at home**, then release **X**. Releasing **X**, squeezing a grip,
or losing tracking interrupts the return. The grippers keep their current positions. Press **A** at home to start
the next take, or return home while the episode is still recording and then press **A** to include the return in it.

New to the headset? **Simulation** runs the same thing with a MuJoCo model of the bench: nothing moves. **Dry run** runs
the real arm program with simulated arms, to check the whole chain without contacting an arm.

## 5. Check, review, export

1. **Episodes** tab → pick the session → **Run checks**. Each episode gets *pass* or *needs review* with the
   reasons (camera drops, timing, missing labels...).
2. Look at each episode (preview, timing plot), fix labels, **Accept** or **Reject**.
3. **Jobs** tab → **Export accepted episodes**: name the dataset, pick the image size (640 × 360 is a good default
   for policy training), start. The dataset lands in `data/lerobot/<name>/`.

## When something is off

**System** tab: every check of the PC, with a button for each fix the app can make. Then
[troubleshooting.md](troubleshooting.md).

## Using it at the bench

**Where it runs.** On the bench PC: the one with the camera (USB 3), the arms' network switch (wired port) and the
Quest (USB-C). Install it there. A copy on a laptop can't reach the hardware.

**From your own laptop.** You need an account on the bench PC. On the laptop, run

```bash
ssh -N -L 8440:localhost:8440 <your user>@<bench PC>
```

and open http://localhost:8440/. The app only listens on the bench PC itself, so this forward is the way in.
Someone still has to be at the bench: the operator wears the headset, and someone stands at the arm power switch.
The one-time System setup asks for a password on the bench PC's own screen; over SSH the app shows the command to
run instead.

**Several people, one bench.** Only one copy of the app can use the camera and the arms at a time. Simplest: one
shared install on the bench PC that everyone opens from their own laptop. If each person installs their own copy,
quit the other copy first (System tab → **Quit application**) and point everyone's data at one shared folder
(below).

**Where the data goes.** On the bench PC, inside the repository folder. Data is never uploaded to GitHub.

| What | Where |
|---|---|
| Raw episodes (lossless video, frame times, labels, check results) | `data/raw/<date>/session_HHMMSS/episode_NNNN/` |
| LeRobot datasets | `data/lerobot/<name>/` |
| Arm logs | `teleop/runs/<date>/` |

Raw recording takes about 1.2 GB per minute at the default 1080p 15 fps (both eyes, lossless), about 2.6 GB with the ZED SDK's SVO2 as well. To keep data on a shared or bigger drive,
set `paths.data_raw` and `paths.data_lerobot` in `configs/rig.yaml` ([configuration.md](configuration.md)). Back up
`data/raw` and `teleop/runs` together.

**Moving the bench to another PC.**

1. Install on the new PC (**Install.sh**).
2. Go through the **Bench** tab there: bench settings are stored per PC (`configs/local/`, `configs/rig.yaml`), not
   in the repository.
3. In the headset, allow USB debugging for the new PC.
4. Recordings stay on the old PC. Exported datasets can be copied anywhere. To check or export raw episodes on
   another PC, copy each session folder (`data/raw/...`) together with the matching arm logs (`teleop/runs/...`):
   the checks pair each episode with its arm log by time. Easiest: check and export on the PC that recorded them.

**What a dataset contains.** LeRobot v3.0, readable with LeRobot's own `LeRobotDataset`: joint state and action for
each arm, both camera eyes as video, and per-step flags for repeated frames and exposure times. Depth and point
clouds are not in the dataset; depth is computed separately with the `depth` command ([data_format.md](data_format.md)).
The lossless recording stays in `data/raw`, so a dataset can be exported again at another image size.
