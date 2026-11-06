#!/usr/bin/env bash
# One-time system setup for the collection app. Needs root: the app and the installer run it through pkexec,
# so Ubuntu's own password dialog asks first. It does only these fixed things, and only the ones still missing:
#   1. the udev rule that lets a normal user read ZED cameras (Stereolabs USB vendor 2b03)
#   2. system packages: ffmpeg, v4l-utils, adb (with its USB rules), power-profiles-daemon
#   3. with --network <interface>: a NetworkManager connection giving that wired port 192.168.1.10/24, no gateway
# Options: --dry-run (print what it would do, change nothing; works without root).
set -euo pipefail

dry=0
network_if=""
while (($#)); do
  case "$1" in
    --dry-run) dry=1 ;;
    --network)
      shift
      network_if="${1:-}"
      if [[ ! "$network_if" =~ ^[A-Za-z0-9._-]{1,15}$ ]] || [[ ! -e "/sys/class/net/$network_if/device" ]]; then
        echo "not a wired network interface: '$network_if'" >&2
        exit 2
      fi
      ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

if ((!dry)) && ((EUID != 0)); then
  echo "run me through pkexec (or with --dry-run)" >&2
  exit 1
fi

run() {
  echo "+ $*"
  if ((!dry)); then "$@"; fi
}

changed=0
# Our own file: the ZED SDK installer rewrites 99-slabs.rules without the video device line.
rule=/etc/udev/rules.d/98-zed-video.rules
want='SUBSYSTEM=="video4linux", ATTRS{idVendor}=="2b03", MODE="0666"
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="2b03", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="2b03", MODE="0666"'
if [[ ! -f "$rule" ]] || ! grep -q 'video4linux.*idVendor}=="2b03"' "$rule"; then
  echo "== ZED camera access (udev rule $rule)"
  if ((dry)); then echo "+ write $rule"; else printf '%s\n' "$want" > "$rule"; fi
  run udevadm control --reload-rules
  run udevadm trigger --subsystem-match=video4linux --subsystem-match=usb --subsystem-match=hidraw
  changed=1
fi

packages=()
for pair in ffmpeg:ffmpeg v4l2-ctl:v4l-utils adb:adb powerprofilesctl:power-profiles-daemon; do
  command -v "${pair%%:*}" >/dev/null 2>&1 || packages+=("${pair#*:}")
done
if ((${#packages[@]})); then
  echo "== system packages: ${packages[*]}"
  run env DEBIAN_FRONTEND=noninteractive apt-get update -q
  run env DEBIAN_FRONTEND=noninteractive apt-get install -y -q "${packages[@]}"
  changed=1
fi

if [[ -n "$network_if" ]]; then
  echo "== arm network on $network_if (192.168.1.10/24, no gateway)"
  if nmcli -t -f NAME connection show | grep -qx widowxai-arms; then
    run nmcli connection delete widowxai-arms
  fi
  run nmcli connection add type ethernet ifname "$network_if" con-name widowxai-arms ipv4.method manual \
    ipv4.addresses 192.168.1.10/24 ipv4.never-default yes ipv6.method disabled connection.autoconnect yes
  run nmcli connection up widowxai-arms
  changed=1
fi

if ((changed)) && ((dry)); then
  echo "dry run: nothing was changed"
elif ((changed)); then
  echo "done"
else
  echo "nothing to do: everything is already set up"
fi
