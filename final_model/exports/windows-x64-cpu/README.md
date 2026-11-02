# Vial inspector — Windows x64 CPU

This is a separate CPU-only Windows export of the compression-aware RT-DETR
detector. It does not use or replace the GB10 CUDA export. It requires a
64-bit Windows installation and Python 3.12; it does not require NVIDIA
hardware, CUDA, TAO, or the training checkpoint.

The archive contains the ONNX model, inference code, fixed-camera calibration,
and package instructions. The Python dependencies are installed on the Windows
computer from PyPI, so the first setup needs internet access. This export was
validated for numerical parity on selected synthetic validation frames using
the current GB10 reference. At confidence 0.1, matched detections on the
selected frames had at least 0.989 box IoU. CPU/GPU arithmetic can differ for
detections close to the threshold: one selected frame had one additional
category-2 detection on GB10 at the default 0.25 cutoff. See
`model/model.json` for the per-frame comparison. CPU speed depends on the
computer; this detector may run substantially slower than the GB10 version.

## Setup

1. Install 64-bit Python 3.12 from <https://www.python.org/downloads/windows/>.
   In the installer, enable **Add python.exe to PATH**.
2. Extract `rtdetr-v3-006-final-windows-x64-cpu.zip`.
3. Open PowerShell in the extracted `windows-x64-cpu` folder.
4. Create an isolated environment and install the pinned runtime packages:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   If PowerShell blocks environment activation, skip the activation line and
   use `.\.venv\Scripts\python.exe` in place of `python` below.

## Run on a video

Use an input video recorded from the same 640×480 camera/tray view as the
included calibration. A different camera or resolution needs matching
calibration and may need retraining; resizing an unrelated camera view does
not make the calibration valid.

```powershell
python run.py --video "C:\path\to\inspection.mp4" --output ".\results" --threshold 0.25
```

The output folder must not already exist. The script writes:

- `inspection.mp4` — annotated video.
- `frame-results.jsonl` — per-frame occupancy, defects, and order status.
- `summary.json` — run settings and frame counts, not an accuracy report.

Category 1 is vial presence, 2 is tilted cap, and 3 is missing cap. The default
confidence threshold is 0.25 and is not a production-calibrated value.

To run a second video, choose a new output directory. To use another calibrated
camera, provide its calibration JSON with `--calibration path\to\calibration.json`.

## Model and sharing notes

The model was trained from the team's synthetic vial data. Its performance on
real camera imagery has not been established. Data-sharing terms for the
provided digital-twin source are still unknown; this bundle contains no
training data or checkpoints. Confirm the model and underlying asset terms
before redistributing this bundle outside the team.
