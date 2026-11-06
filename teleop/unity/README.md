# Unity status

The first Windows implementation uses the article's WebXR approach and does not require Unity. The Quest Browser reads controller poses directly and reaches the PC through `adb reverse`.

Keep this directory reserved for a future Unity client only if WebXR cannot provide a required feature. Any Unity transport must use the same packet schema, latest-state semantics, sequence/timestamp fields, stale watchdog, and replay tests as the WebXR baseline.

