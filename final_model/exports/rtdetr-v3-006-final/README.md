# Compression-aware vial inspection model

This bundle runs the final epoch-327 model from training-v3-006. It contains the TorchScript detector, required CUDA operator and metadata, inference code, G01–G30 slot calibration, a sample video and its generator truth, and dependency instructions. No TAO repository, training checkpoint, Docker or SLURM is needed for inference.

## Supported environment

The packaged operator targets Linux ARM64/aarch64, NVIDIA GB10 (compute capability 12.1), Python 3.12, PyTorch 2.11.0 with CUDA 13.0. It was tested on driver 580.178.04. A compatible NVIDIA driver and CUDA runtime must already be installed. Python packages require internet access to install; wheels and the Python/GPU environment are not bundled. The archive is not a universal CPU, Windows, x86-64 or other-GPU export: those targets need a separately built operator and verified export.

## Unpack

```bash
tar -xzf rtdetr-v3-006-final.tar.gz
cd rtdetr-v3-006-final
```

On the existing training machine, reuse its working interpreter (substitute the absolute repository path):

```bash
PYTHONNOUSERSITE=1 /absolute/path/to/repository/new-model/.envs/tao-vcn/bin/python -s run.py --output sample-results
```

On another compatible GB10 machine with Python 3.12 available, create an environment without administrator access:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -r requirements.txt
PYTHONNOUSERSITE=1 .venv/bin/python -s run.py --output sample-results
```

The launcher verifies bundled file hashes, Python/PyTorch/CUDA versions, and GPU architecture. The runtime loads the operator before TorchScript and disables JIT optimization to preserve the export's verified numerical behavior. Keep all three files in model/ together.

## Inspect your video

```bash
PYTHONNOUSERSITE=1 .venv/bin/python -s run.py \
  --video /absolute/path/to/video.mp4 \
  --output my-video-results \
  --threshold 0.25
```

Use the working interpreter instead of .venv/bin/python when reusing the training environment. Output must be a new directory. Paths resolve relative to your current directory; bundle assets resolve relative to run.py, so it also works when launched elsewhere. The default threshold is the existing test cutoff 0.25, not a production-calibrated operating point. Confirmation defaults to three consecutive decoded frames, which means a different time delay at a different video frame rate.

Input must decode to exactly 640×480 and use the same fixed camera/tray layout as the supplied calibration. Arbitrarily resizing another camera view does not establish correct G-slot mapping. Supply --calibration /path/to/calibration.json for a separately calibrated matching view; retraining may be needed for real-camera imagery. S tray slots are excluded.

## Outputs and meaning

- inspection.mp4: annotated video. Red boxes are raw defect candidates above the cutoff; they can appear before temporal confirmation.
- frame-results.jsonl: per-frame confirmed defects with G-slot IDs, occupancy, unknown/empty slots, fill-order gaps and complete/incomplete status.
- summary.json: run settings and frame counts. It is not an accuracy report.

Any empty G slot makes the order incomplete. All 30 occupied slots make it complete, independently of cap defects. Empty slots behind the fill front are gaps; trailing empty slots can be normal filling. Category 1 is vial presence, 2 tilted cap, 3 missing cap. Detector input is RGB float32 [1,3,480,640] in [0,1]; output is Nx6 [x0,y0,x1,y1,score,category_id] in native pixels. run.py handles decoding and this conversion.

The sample contains defects G01/G02 (missing caps), G09 (~21.49° tilt), G21 (~21.56° tilt). The first three appear at 0.8 seconds and G21 at 4.0 seconds; all slots are full from 4.8 seconds. Model predictions can differ. See examples/ground-truth.json.

## Provenance and limits

Model source: final_model/training-v3-006/final_epoch_327.pth, trained on an approximately 50/50 mix of clean and decoded mp4v synthetic images. This is a frame detector followed by temporal rules. Validation/test results concern synthetic views only; real-camera performance is not established. Each invocation starts a new temporal state; continuous multi-order streams need explicit order-boundary handling.

manifest.json records artifact SHA256 hashes and the source checkpoint hash. model/rtdetr-v3-006-final.json records four-frame export parity checks. bundle-test.json records the unpacked sample run. The original .pth optimizer/training state is intentionally retained in the training directory; this bundle is for inference, not resuming training.
