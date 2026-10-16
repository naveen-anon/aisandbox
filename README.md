# AI Sandbox Trainer

On-device, fully offline training APK. Training runs inside an OS-enforced
sandboxed process, not just a "logically separate" class.

## Why this is a real sandbox, not a label

`TrainingService` runs with `android:isolatedProcess="true"` in process
`:training_sandbox`. Android assigns that process a throwaway, per-instance
UID from the `isolated` UID range. Consequences, enforced by the kernel /
SELinux policy, not by this app's code:

- **No network.** Isolated UIDs are denied raw sockets regardless of any
  `INTERNET` permission — and this app doesn't even request that permission,
  anywhere, so there's nothing to misuse even at the main-process level.
- **No access to this app's files, prefs, or ContentProviders.** The
  isolated process is not "this app" as far as DAC permissions go.
- **No permissions.** Isolated UIDs are never granted runtime permissions
  (camera, storage, etc.) even if the main app holds them.
- **Dies cleanly.** Unbinding kills the process; no leftover state.

The only channel in or out is a `Messenger`/`Binder` connection
(`TrainingProtocol.kt`) carrying explicit `Bundle`/`byte[]` payloads:
model bytes in, labeled samples in, progress + checkpoint bytes out.
`SandboxController` (main process) is the single chokepoint allowed to
talk to it — don't bind to `TrainingService` from anywhere else.

## Architecture

```
MainActivity (main process)
  -> SandboxController             [only class allowed to bind the sandbox]
       -> bindService() ---Binder--> TrainingService  (:training_sandbox, isolated)
                                          -> SandboxTrainer
                                               -> TFLite Interpreter
                                                    signatures: train / infer / save / restore
```

Training uses TensorFlow Lite's on-device training pattern (same approach
as Google's official Model Personalization sample): a frozen MobileNetV2
feature extractor + a small trainable Dense head. Only the head updates
on-device — this is what makes training fast enough to run on a phone CPU
instead of needing the full backbone's gradients.

## Building

You cannot build this on-device in proot (Android SDK build-tools don't run
there). Push to GitHub and let `.github/workflows/android-build.yml` build
it — matches your existing CI-based workflow. APK comes back as a workflow
artifact.

```
git init && git add -A && git commit -m "init"
git remote add origin <your repo>
git push -u origin main
```

## Before first build: generate the base model

The repo does **not** include `app/src/main/assets/base_model.tflite` —
that's a real trained-weights binary (MobileNetV2 + ImageNet weights),
too large and not something to fake. Generate it once on a normal machine:

```
pip install tensorflow==2.16.1
python tools/build_base_model.py
```

This writes `app/src/main/assets/base_model.tflite`. Commit it (or better:
Git LFS — it'll be tens of MB) before pushing.

If you change `NUM_CLASSES`, `IMAGE_WIDTH`, or `IMAGE_HEIGHT`, update both
`tools/build_base_model.py` and
`app/src/main/java/com/naveen/aisandbox/ipc/TrainingProtocol.kt`
(`ModelSpec`) to match — the Kotlin side builds tensors against those
constants and will mismatch the graph's expected shapes otherwise.

## What's stubbed / what you still need to wire

- `MainActivity.btnAddSample` currently pushes a zeroed dummy image —
  wire it to your camera (`CAMERA` permission is already declared) or
  gallery picker, resize to `IMAGE_WIDTH x IMAGE_HEIGHT`, convert to raw
  RGB `ByteArray` (see `SandboxTrainer.addSample` for the exact
  normalization it expects).
- Checkpoint **restore** on next launch (load a previous `.ckpt` from
  `CheckpointStore` back into a fresh `SandboxTrainer` via the `restore`
  signature) — `save`/the write path is implemented, `restore` call-site
  isn't wired into the UI yet.
- Export: if you want the trained model to leave the device at all, that
  has to be a deliberate `ACTION_CREATE_DOCUMENT` / share-sheet flow in
  the main process — don't add it to the sandbox process, don't add
  `INTERNET` back to make it "easier."

## Threat model notes

- This protects against: a buggy/malicious training loop or a poisoned
  training sample trying to read other app data, hit the network, or
  escalate permissions.
- This does **not** protect against: a malicious *base model file* itself
  containing a custom TFLite op that exploits an Interpreter native-code
  vulnerability — that's a TFLite-runtime CVE surface, not something
  process isolation alone closes. Keep `tensorflow-lite` dependency
  versions current.
- `isolatedProcess` services on Android can still consume CPU/battery
  freely — this is resource sandboxing for *data/permissions*, not a
  cgroup CPU/memory cap. Add `WorkManager` constraints (already a
  dependency) if you want training to only run while charging/idle.
