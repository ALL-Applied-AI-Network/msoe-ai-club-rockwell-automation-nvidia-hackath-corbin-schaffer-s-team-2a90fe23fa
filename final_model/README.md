# Video-based G-tray inspection — final_model

This directory contains the full-frame video pipeline and synthetic video datasets. The dataset-v3 RT-DETR training run has completed, and a verified TorchScript export is available at `models/rtdetr-v3.ts`. Earlier dataset-generation sections describe the preparation history. The existing paired classifier is preserved.

## Data and model storage

See [`data/README.md`](data/README.md) for the data-handling policy, provenance register, sample, and current full-data access status. The complete datasets, checkpoints, and compiled model exports are local artifacts and are excluded from Git; do not add them with `git add -f`. Existing dataset directories are left in place for compatibility with the scripts below.

For a separate Windows x64 CPU inference package, see [`exports/windows-x64-cpu/README.md`](exports/windows-x64-cpu/README.md). It is independent of the GB10 CUDA export; regenerate it with `python final_model/export_windows_cpu.py` on the verified GB10 training host. The generated ZIP is intentionally ignored by Git because it contains a large model; transfer it to teammates only through approved storage.

## Architecture

`video → full-frame detector → G01–G30 assignment → temporal confirmation → defects, gaps, order status`

`inspect_video.py` decodes frames and accepts either a trained TorchScript detector or external per-frame predictions. Detector contract: RGB float BCHW in [0,1], native 640 × 480 input, tensor Nx6 containing xyxy boxes, score, category ID (1 vial, 2 tilted cap, 3 missing cap). The trained RT-DETR export `models/rtdetr-v3.ts` now matches that contract, with preprocessing and native-coordinate postprocessing included. The script does not run ground truth as a model.

Slot assignment uses the existing fixed synthetic-camera calibration, with G priority. Recalibrate if the camera or tray geometry changes. Temporal logic confirms occupancy/defects after three observed frames by default. Three missed observations clear occupancy. Brief absent evidence is unknown and cannot declare an order complete. All 30 G slots occupied means complete, regardless of cap quality. Empty slots before a later occupied slot are gaps; trailing empty slots make the order incomplete without a gap. S slots are ignored. Each video starts with a fresh tracker; call `reset()` on a new order/tray in a continuous application.

## Dataset generation

From the repository root, using the previously prepared native environment:

```bash
PYTHONNOUSERSITE=1 new-model/.envs/tao-vcn/bin/python -s final_model/generate_dataset.py \
  --output final_model/dataset-v2 --train-defect-probability 0.5 --eval-defect-probability 0.1
```

The default `final_model/dataset-v2` must not already exist. The original version is retained under `final_model/dataset`. Generation is complete for this implementation; see `dataset/status.json`, `dataset/validation.json` and `generation.log`.

Nine sequences: normal filling, skipped G12, and cap defects (tilted G08, missing cap G18), each with a different seed in train/validation/test. Each has eight rendered placement states, four held frames per state at 5 FPS: 32 frames / 6.4 seconds. All source vial assignments remain fixed within a sequence; images are not shuffled random scenes. Normal/cap-defect videos reach full occupancy; gap videos remain incomplete.

The renderer snapshot `tools/render_video_states.py` is independent of the existing renderer. It reads the original USD and changes only in-memory session state. Existing source files and datasets remain unchanged. The helper launcher and cap-label/annotation utilities are reused by import.

Outputs:

- `videos/*.mp4`: full-frame synthetic video.
- `frames/<sequence>/*.png`: lossless frame-level detector inputs.
- `annotations/{train,validation,test}.json`: COCO boxes with `slot_id`, sequence/frame IDs, timestamps, and order ground truth.
- `sequences/*.json`: every frame's slot occupancy, defects, fill-order gaps, complete/incomplete state and timed placement events.
- `sequences/*.oracle-predictions.json`: labeled **ground-truth replay fixtures**, never model predictions.
- `calibration.json`: fixed camera G-slot anchors.
- `sequence-index.json`: seeds, split assignments and scenarios.

All frames and held-frame copies from a sequence stay in the same split. Seeds differ, but original vial geometry families are shared: this is not an independent-defect-family benchmark. These videos depict discrete placement changes, not robot animation, vial travel, sensor-realistic noise or integrated motion blur. Repeated held frames are not independent samples. Nine short videos are a pipeline-development dataset; larger varied sequences and real-camera evaluation are still needed before production use.

## Run video inspection

With a future trained detector:

```bash
PYTHONNOUSERSITE=1 new-model/.envs/tao-vcn/bin/python -s final_model/inspect_video.py \
  --video input.mp4 --calibration final_model/dataset/calibration.json \
  --model detector.ts --output final_model/inference-run
```

An external detector can supply JSON records with `frame_index`, `bbox` (native xywh), `category_id`, and `score` via `--predictions`. Input slot IDs are ignored; boxes are mapped to G coordinates. To render a test video using labels instead of a model, pass either its `sequences/<id>.json` file or its `sequences/<id>.oracle-predictions.json` file via `--ground-truth`:

```bash
python final_model/inspect_video.py \
  --video final_model/dataset-v3/videos/test_fill_000.mp4 \
  --calibration final_model/dataset-v3/calibration.json \
  --ground-truth final_model/dataset-v3/sequences/test_fill_000.json \
  --output final_model/dataset-v3/qa/test_fill_000-ground-truth
```

The initial confidence cutoff is 0.1 and must later be calibrated on validation sequences, prioritizing defect recall as requested. No threshold is chosen using the test split.

Outputs are annotated `inspection.mp4`, per-frame `frame-results.jsonl` and `summary.json`. Outputs distinguish trained-detector execution, external prediction replay, and ground-truth replay. Zero-detection frames are decoded and reported, not silently skipped.

## Verification

```bash
python final_model/tests/test_temporal.py
PYTHONNOUSERSITE=1 new-model/.envs/tao-vcn/bin/python -s final_model/validate_dataset.py --dataset final_model/dataset
```

Validation checks decoded videos, stable source identities, frame timestamps, class/slot geometry, gap/completion ground truth, whole-sequence splits, and end-to-end ground-truth replay through the video pipeline. It writes `dataset/validation.json` and a visual contact sheet. Replay confirms software wiring only; it does not demonstrate detector performance or remove the need to train a detector.

## Delivered dataset verification

All nine videos decode successfully: 288 total frames (96 per split) from 72 rendered scene states. Each split contains normal, gap and cap-defect sequences. Stable identities, timestamps, box/slot labels and full sequence separation passed checks. All nine ground-truth replay tests produced the expected final occupancy, G12 gap and G08/G18 defect outputs. Visual contact sheet: `dataset/qa/contact-sheet.jpg`. This verifies the synthetic labels and video pipeline, not a learned detector. No training was started.

## Version 2: defect probabilities by split

New video generation defaults to `final_model/dataset-v2` with a **50% per-placed-vial defect probability for training** and **10% for validation and testing**:

```bash
PYTHONNOUSERSITE=1 new-model/.envs/tao-vcn/bin/python -s final_model/generate_dataset.py \
  --output final_model/dataset-v2 \
  --train-defect-probability 0.5 --eval-defect-probability 0.1
```

Version 1 in `dataset/` is preserved. Version 2 generation is tracked in `dataset-v2/status.json` and `generation-v2.log`. To regenerate, select a different output path. No training is launched.

Probability applies to each distinct placed vial, not each video frame or an entire tray. A separate seeded sampler chooses a cap condition once per slot and retains it throughout the sequence. Sampled defects are split between tilted and missing caps, with tilt sampled from 2–25 degrees. A skipped/empty slot is not counted as a defective present vial. The actual finite-sample percentage can differ from the requested probability; `defect-rate-summary.json` reports unique placed-vial counts and rates, without counting repeated frames as independent vials.

Version 2 scenarios are `fill`, `gap` (skip G12), and `fill_pause` (hold the partially filled tray longer). All scenarios use the requested split-specific cap probabilities; `fill` means normal placement order, not guaranteed defect-free caps. Whole sequences stay within one split. Validation/testing have lower probabilities and are not balanced by copying frames across splits.

Validate the new version with:

```bash
PYTHONNOUSERSITE=1 new-model/.envs/tao-vcn/bin/python -s final_model/validate_dataset.py \
  --dataset final_model/dataset-v2
```

This checks persistent condition labels, actual class proportions and all video/temporal checks, while preserving the original version's validator compatibility.

Version 2 is now generated and validated: nine videos / 288 frames. Actual unique placed-vial rates are train 50/89 (56.2%), validation 2/89 (2.2%), test 11/89 (12.4%). Requested probabilities are 50%, 10%, 10%; random finite-sample counts need not match exactly. All nine temporal replays passed, including the sampled defect-slot labels. See `dataset-v2/validation.json`. No training was started.

## Expanded training dataset (version 3)

The nine-video starter dataset is not sufficient for a useful training/evaluation decision: it has only three training trajectories (24 rendered states) and two validation defects. An expanded dataset is being prepared in `dataset-v3`: 60 train, 15 validation and 15 test sequences, 720 rendered states / 2,880 video frames. Cap defect probabilities remain 50% for training and 10% for evaluation. This is a proposed baseline scale, not a universal minimum or a guarantee of real-camera performance.

Version 3 varies seeded fill schedules, skipped G-slot positions and light intensity (0.8–1.2 × baseline), while keeping each video's vial identities, defect conditions and camera fixed. All frames from one sequence remain in one split. `annotations/*-keyframes.json` omit held-frame copies and repeated placement states within each sequence; use these for detector training, while keeping complete videos for temporal evaluation. Source geometry families still overlap across splits.

Track `preparation-v3/status.json` for generation plus validation, `dataset-v3/status.json` for the active sequence, and `preparation-v3/assessment.json` for the data-size decision. The pipeline automatically validates all videos and temporal-label replays and checks per-class keyframe coverage. It does **not** launch training. A full-frame detector runtime/configuration smoke test is still needed after data preparation.

## Dataset-v3 detector training

Dataset preparation passed: 398 training, 100 validation and 99 held-out test keyframes, grouped into 60/15/15 sequences. Defect rates: 50.9%/8.3%/11.9%. RT-DETR imports and a one-step GPU train/validation smoke test passed in the existing native conda environment.

A single full run was launched in `training-v3-001`: ResNet-50 initialized with official ImageNet weights, 50 epochs, batch 4, FP32, AdamW lr 1e-4 (backbone 1e-5), LR drops at epochs 35/45, validation every epoch, and best checkpoint selected by validation mAP50. Class IDs are 1=vial, 2=tilted cap, 3=missing cap; the native head has four slots including unused category zero. IoU cropping is disabled to retain tray context; mild photometric augmentation is enabled.

Track `training-v3-001/run-status.json`, `train.log`, and CSV metrics. This trains the full-frame detector, not a recurrent network; the temporal G-slot video pipeline consumes its detections afterward. The held-out test split is not used in training. Export/adaptation to the video inference contract and test-video evaluation remain subsequent steps.

## Run the trained TorchScript detector

The adapter/export is implemented in `detector_adapter.py` and `export_detector.py`. The exported model uses the best epoch-27 checkpoint `training-v3-001/model_best_026.pth` (zero-based checkpoint filename).

```bash
PYTHONNOUSERSITE=1 new-model/.envs/tao-vcn/bin/python -s final_model/inspect_video.py \
  --video final_model/dataset-v3/videos/test_fill_000.mp4 \
  --calibration final_model/dataset-v3/calibration.json \
  --model final_model/models/rtdetr-v3.ts \
  --output final_model/my-video-test
```

Choose a new output directory for each run. Open its `inspection.mp4` and `frame-results.jsonl`. A completed smoke preview is already in `video-test-v3-001/`, generated with the learned detector (not oracle labels). All 32 frames were decoded; this was a functional inference test, not a scored video-accuracy evaluation.

Keep these three files together when moving the model:

- `models/rtdetr-v3.ts`: TorchScript graph.
- `models/rtdetr-v3.ops.so`: compiled TAO CUDA inference operator, built for the verified GB10 environment.
- `models/rtdetr-v3.json`: input/output contract, operator checksum, ABI information and parity results.

The video script verifies the operator checksum and registers it before loading TorchScript. Use the existing `tao-vcn` environment with matching PyTorch/CUDA ABI; the artifact is CUDA-dependent, not a standalone CPU model. JIT graph optimizations are disabled in the runner because they changed numerical ordering during export checks. No original TAO source modifications were needed.

Input is one RGB float32 frame `[1,3,480,640]` in `[0,1]`. The adapter resizes to the trained 640 × 640 size with antialiasing and uint8-grid quantization, runs the detector, applies sigmoid/top-K and converts relative boxes back to native 640 × 480 coordinates. Category zero is removed; categories 1/2/3 retain their original IDs. No decision threshold is baked into the artifact; `--threshold` remains adjustable. The PIL-based dataset resize and tensor resize can differ slightly at pixel quantization boundaries; exact zero-error parity was verified between TorchScript and eager execution using identical adapter preprocessing.

To export again without overwriting the verified artifact:

```bash
PYTHONNOUSERSITE=1 new-model/.envs/tao-vcn/bin/python -s final_model/export_detector.py \
  --checkpoint final_model/training-v3-001/model_best_026.pth \
  --spec final_model/training-v3-001/spec.yaml \
  --output final_model/models/rtdetr-v3-new.ts
```

Export strictly loads all checkpoint parameters, checks four validation frames against eager output and TAO's native postprocessor, rejects wrong input dimensions, and verifies save/reload parity. Adapter unit tests cover native box scaling, category IDs and shape rejection. See `models/rtdetr-v3.json` for the verified artifact's report. No training is started by exporting or inspecting a video.

## Additional 50-epoch continuation

`training-v3-002` resumes `training-v3-001/model_best_026.pth`, which contains 27 completed epochs, for **50 additional epochs** (target total 77). Model tensors, optimizer and scheduler state are restored. Original milestones at total epochs 35/45 are preserved; no new test data is used. Best/latest and an explicit final-epoch checkpoint are retained for the new run. Existing checkpoints and TorchScript exports remain unchanged.

The source checkpoint was saved after validation but before epoch-end hooks. Its completed-batch loop bookkeeping needed a boundary correction in `training-v3-002/resume_epoch_boundary.pth`; the original file was preserved. `resume-state-verification.json` confirms that every model tensor, optimizer state and scheduler state is identical. The failed initial resume log is preserved separately. Live status/log: `training-v3-002/run-status.json` and `training-v3-002/train.log`.

## Fresh comparison run: training-v3-003

A new detector is training independently on dataset-v3 video-derived keyframes. It starts from the same ImageNet ResNet-50 backbone with a newly initialized detector head, not from either trained checkpoint. Seed 43 differs from the original seed 42; architecture, 50-epoch budget, batch size 4, optimizer and train/validation splits are held constant. The 99 test keyframes remain unused during training. Best validation-mAP50 and final checkpoints will be saved separately.

Track `training-v3-003/run-status.json` and `train.log`. `comparison-plan.json` records the setup and dataset annotation checksums. This independent run measures seed variation; it does not implement a false-alarm-specific loss or promise fewer false positives. Compare both models at the same calibrated cutoff and on identical held-out videos before deciding which to deploy. Original model artifacts and exports remain unchanged.


### Compression-aware continuation: training-v3-006

`dataset-v3-compression/` contains 398 clean and 398 decoded-MP4 training keyframes from the original 60 training sequences. Labels and sequence splits are preserved. Clean image files link to dataset-v3; compressed variants are saved losslessly after decoding its existing mp4v training videos. No test frames enter this dataset. There are 100 clean and 100 compressed validation keyframes.

The run resumes the source checkpoint of `models/rtdetr-v3.ts` (`training-v3-001/model_best_026.pth`) for 300 additional epochs, batch size 4, seed 42, with the original optimizer/scheduler. `--max-train-batches 99` retains the original per-epoch update budget: 396 samples shuffled from 796 candidates each epoch, approximately 50/50 clean/compressed. It does not process every candidate each epoch.

Clean COCO validation still runs every epoch and selects the best checkpoint. `paired-validation.jsonl` separately records clean/compressed slot recall and false-alarm counts at fixed cutoff 0.25 at epoch 28, every tenth epoch, and the final epoch. This is validation, not held-out testing, and is distinct from COCO metrics. Best and final checkpoints remain separate; inference exports/defaults are unchanged.

Preparation: `prepare_compression_dataset.py`. Training additions: `--dataset-root`, `--max-train-batches`, `--paired-validation`. The resumed one-batch smoke passed in `training-v3-006/smoke/`. Durable progress: `training-v3-006/run-status.json`, `train.log`, and `metrics/`. The original clean-only long run is `training-v3-005`.


### Inference export bundle

`exports/rtdetr-v3-006-final.tar.gz` packages the final compression-aware model, its CUDA operator, standalone inference code, camera calibration, example video and instructions. See `exports/rtdetr-v3-006-final/README.md`. Tested after extraction outside the repository with the existing Python environment; targets Linux ARM64 GB10, Python 3.12, PyTorch 2.11.0/CUDA 13.0. A fresh dependency installation on another host has not been tested.


### H.264 inspection output

`inspect_video.py` writes H.264 MP4 with yuv420p pixels and fast-start metadata using PyAV's libx264 encoder (CRF 18, fast preset). Install the encoder dependency in the same Python environment: `python -m pip install av` (verified with av 19.0.2 on ARM64). This avoids the unavailable OpenCV hardware H.264 encoder. Commands and output filenames are unchanged; overlays are encoded after inference, so input pixels and predictions are unaffected. Older saved videos and standalone export bundles retain their original codec until regenerated. `summary.json` records the output codec and pixel format.


### Missing-vial postprocessing

`inspect_video.py` now accepts `--presence-threshold` (default 0.25) separately from the cap `--threshold`, plus `--absence-frames` (default 3). Confirmed fill-order gaps and sustained disappearance of previously occupied vials create postprocessing missing-vial defects. Trailing empties remain incomplete only; uncertain slots are yellow. `temporal.VideoInspector` retains its previous shared-threshold behavior when presence_threshold is omitted, for API compatibility. Its output now includes cap_defects, missing_vial_defects, missing_slots and occupied_slots; the combined defects list includes missing_vial entries with category_id null. Cap-only evaluators should read cap_defects.

Review the learned-model H.264 examples and validation results in `missing-vial-examples-006/README.md`. No model weights or previously packaged export bundles were changed. Eight temporal tests cover these decisions.


### Permanent defect summary

Every new inspect_video output includes a bottom-left Actual / Model / Confirmed summary. Actual is independent generator/annotation truth, including cap defects and fill-order gaps; Model shows raw above-threshold cap candidates and postprocessed missing-vial alerts; Confirmed shows temporally confirmed cap/missing-vial alerts. Lists use G-slot IDs. Text wraps to fit and uses a shaded panel.

For dataset videos, labels are automatically read from the adjacent dataset sequences/<video-stem>.json. Use `--labels /path/to/sequence.json` to supply independent labels explicitly while still using `--model`. Frame count and FPS are checked when declared. Labels are displayed and logged after inference; they are never fed into detection or tracking. If labels are absent (or a frame is not labeled), the summary says Actual: unavailable. Ground-truth replay remains a separate, explicitly labeled mode.

`frame-results.jsonl` now records actual_defects and ground_truth_available. `summary.json` records the label source and legend. The previous red/green/yellow boxes and H.264 output remain. Example: defect-summary-006/test_gap_004/inspection.mp4. A labeled/unlabeled comparison on 32 frames verified identical model decisions with and without independent labels. Existing saved videos and export bundles must be regenerated to include this overlay.


### Fresh one-vial-at-a-time tests

`one-by-one-test-006/video-index.md` links ten fresh source and inspected videos, each 31 seconds at 5 FPS: empty tray, then exactly one additional vial per second from G01 to G30. Two are defect-free controls; eight sample persistent defects at 10% per vial. Seeds are 6100000–6100009. Source and annotated outputs are H.264. The final 006 model runs at cap cutoff 0.90 and presence cutoff 0.25, with Actual/Model/Confirmed summaries. No model training was performed.

Generation and integrity checks: `generate_one_by_one_test.py`. Renderer option `--one-by-one --variations 31 --sequence-scenario fill` preserves the old eight-state mode when omitted. Scoring/report: `summarize_one_by_one_test.py`; results include per-defect histories and frame-level counts. These are discrete placement states, not physical insertion motion, and use the same camera and geometry families as existing synthetic data.


### Directory rename verification

The project directory is now `final_model/` (formerly `test_2/`). Scripts, saved paths, dataset symlinks, checkpoint path metadata, ignore rules and export manifests were relocated. Checkpoint tensor storage was preserved. Twelve unit tests passed, and the relocated final checkpoint loaded strictly with optimizer and scheduler state retained. The post-rename 32-frame inference produced identical decisions to the pre-rename run; H.264 output and the standalone bundle were verified. Archive manifests/checksums were checked. Details: `directory-rename-report.json` and `rename-verification/verification.json`.
