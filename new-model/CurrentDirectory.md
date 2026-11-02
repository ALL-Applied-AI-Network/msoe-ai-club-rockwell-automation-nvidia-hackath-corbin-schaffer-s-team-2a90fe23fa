# Visual ChangeNet inference plan

This plan covers two separate checkpoint-based inference jobs using the repository's `tao-train-visual-changenet` skill. No training, downloads, container launches, or environment changes are included in this planning task.

## Checklist: what to provide and how

Dataset update: use `GoldenImages/G24.jpg` as the fixed golden reference and digital-twin vial identities/placements as the synthetic classify label source. See [dataset.md](dataset.md) for remaining work. `[-]` marks items this plan will address; they are pending, not completed. Items 6 and 10 are only partly covered.

Complete these items for both tasks. Record paths and non-secret choices in the handoff block below. Existing files can stay where they are; provide absolute paths or storage URIs. I will validate inputs and prepare the runnable specs after platform preflight.

- [ ] **1. Choose where the jobs should run.** Choose local Docker on this GB10, remote Docker, SLURM, Kubernetes, or Brev. For local Docker, write `local Docker, 1 GPU`. For a remote target, provide the existing host alias or cluster/context and the GPU allocation you can use. Do not send passwords or private keys. If unsure, write `help choose platform` and identify the machines you can access; I will check the supported options before we select one.

- [ ] **2. Locate the trained classify checkpoint.** Open the output folder from the classify training run and locate the saved full-model checkpoint, such as `model_epoch_*_step_*.pth`. Provide its absolute path and the training run's YAML configuration. If using a `latest` symlink, make sure its target exists. Also record the TAO/container version used for training if known. A backbone-only `.safetensors` file is insufficient. If you have no trained classify checkpoint, write `missing; training needed` rather than selecting an unrelated model.

- [ ] **3. Locate the trained segment checkpoint.** Find the full-model checkpoint from the segment training run and provide its absolute path, training YAML and TAO/container version if known. This is a separate model from the classify checkpoint. If none exists, write `missing; help select a model or train`. I will check the documented pretrained segment option for compatibility and task suitability before proposing a download.

- [-] **4. Prepare classify image pairs.** For every inspection sample, select its inspection image and matching golden/reference image of the same object and view. Keep the lighting conditions expected by the trained model. For a single `SolderLight` example, place the files at `images/inspection/sample001_SolderLight.jpg` and `images/golden/sample001_SolderLight.jpg`. Repeat for each sample using a unique stem. Provide the absolute path to `images/`. If your existing naming/layout differs, provide that path and describe it so I can plan a conversion instead of asking you to rename everything manually.

- [-] **5. Create the classify inference CSV.** Create a UTF-8 CSV with the exact header and one row per inspection/reference pair, as below. Paths in the first two columns are directories relative to the images root. The last column is the shared filename stem without `_SolderLight` or `.jpg`. Use known labels when available; if samples are unlabeled, report that and leave this item pending until I confirm the runtime's accepted inference label values. Do not invent ground truth.

  ```csv
  input_path,golden_path,label,object_name
  inspection,golden,PASS,sample001
  inspection,golden,NO_PASS,sample002
  ```

  The example assumes the corresponding files exist in both directories. Provide the CSV's absolute path and total row count. Include at least two real pairs for the initial classify smoke run because the skill specifies a batch size greater than one.

- [-] **6. Supply classify preprocessing settings.** Provide the original training YAML rather than guessing settings. It should identify the backbone/head, `image_ext`, `input_map`, `num_input`, image dimensions, normalization, `grid_map`, `concat_type` and decision threshold (`model.classify.eval_margin`). If the YAML is unavailable, record the lighting suffixes, extension, image dimensions and any known model settings, and mark the rest `unknown`; checkpoint compatibility must be resolved before launch. If a separate C-RADIO or DINOv3 backbone file was used, provide its path if available. Do not download one yet.

- [-] **7. Prepare segment before/after pairs.** Choose a dataset root and put reference/before images in `A/` and inspection/after images in `B/`. Each pair must have the same filename, for example `A/sample001.png` and `B/sample001.png`. Align the images so corresponding pixels represent the same scene location; do not pair different views or crop them independently. Provide the absolute dataset-root path, pair count and image dimensions. If alignment is not already available, mark that as a preparation task.

- [ ] **8. Create the segment prediction list.** Under the segment root, create `list/test.txt` with one paired image filename per line, for example `sample001.png` then `sample002.png`. Check that every listed filename exists in both `A/` and `B/`. Use `test` as the prediction split name, or record your alternative split basename. Report whether real ground-truth masks exist in `label/`. If they do, provide binary PNG masks with 0 for unchanged and 255 for changed, using the checkpoint's documented naming convention. If they do not, report `no masks`; I will verify the prediction loader's requirements before asking for any additional layout. Do not create dummy masks.

- [ ] **9. Supply segment preprocessing settings.** Provide the segment training YAML with backbone, decoder, class count, `img_size`, normalization and multi-scale settings. If it is unavailable, write `configuration unknown` and provide any model documentation you have. Do not assume the classify settings apply to segment or that the default segment template matches your checkpoint.

- [-] **10. Choose results locations and smoke samples.** Provide a parent directory on the selected compute host where you have write access; proposed subdirectories are `classify/` and `segment/`. You do not need to create them now. Identify a small representative set of real samples for each smoke run, including known unchanged and changed/defective cases where available. State whether you want predictions only or also quantitative evaluation. Evaluation additionally needs trustworthy labels/masks and is a separate action.

- [ ] **11. Make required access available securely.** For local files, ensure your execution account can read the data/checkpoints. For remote storage, identify the URI and an existing credential profile or approved environment-file path. If registry authentication is required, arrange an NGC key through the session environment or an approved env file; report only that it is available. If checkpoints are encrypted, identify the environment variable containing the key. Never paste keys, passwords, tokens or credential-file contents into chat or this document. I will verify presence without printing values and handle approved authentication through stdin.

- [ ] **12. Resolve platform access and approve the concrete launch review.** For local Docker, the earlier check found socket permission denied. Ask the machine administrator to grant your account approved Docker access, or specify an already authorized execution account/host. After access changes, restart the affected session and check `docker info`; do not make the socket world-writable. I will then run the selected platform preflight, check GB10/image compatibility, validate both datasets, and present the exact image, configuration, mounts, GPU allocation and outputs. Confirm that review before any image pull, download or job launch. You do not need to install CUDA/toolkits or change system permissions yourself as part of this checklist.

### Handoff: fill in non-secret values

```text
Platform / host / context: SLURM via SSH alias inno-lab (per Requirements.md)
GPU allocation:
Classify checkpoint path: missing; synthetic dataset preparation then training needed
Classify training YAML path and TAO version:
Classify images root:
Classify CSV path and row count:
Classify lighting / extension / missing settings:
Classify backbone asset path, if available:
Segment checkpoint path:
Segment training YAML path and TAO version:
Segment dataset root and pair count:
Segment prediction split (default test):
Segment image dimensions / alignment status:
Segment masks available (yes/no):
Results parent directory:
Smoke sample names for each task:
Predictions only or separate evaluation requested:
Credential profile / approved env-file path, if needed (no values):
Missing inputs or help needed:
```
## Shared runtime requirements and current blockers

Dataset implementation started: placement export and TAO dataset builder are implemented and fixture-validated; headless rendering has been recovered using the NVIDIA EGL ICD. Three pilot scenes yielded 136 pairs; crop alignment has been corrected using projected mouth/cap geometry and verified on all 136 pilot pairs. Production coverage and label QA remain pending. No production dataset has been released. Keep dataset items `[-]` pending. See [dataset.md](dataset.md).

- Model routing resolves both actions to `tao-train-visual-changenet`; the local metadata pins `nvcr.io/nvidia/tao/tao-toolkit:7.3.0-pyt`. This image tag is separate from the marketplace Git ref `7.0.1`.
- Select the platform before execution. Docker is a candidate, not an assumed selection. The bank also provides Kubernetes, SLURM, Brev, and virtualenv adapters; each needs its own prerequisites and workload compatibility checks.
- Earlier session checks detected an NVIDIA GB10 with driver `580.178.04`. Docker is installed but this session could not access `/var/run/docker.sock`. Resolve daemon access before a local Docker launch.
- Run the selected platform's preflight. The local Docker skill lists driver >=580, CUDA Toolkit >=13.0, NVIDIA Container Toolkit >=1.19.0, and Docker >=20.10 unless model requirements override them. Only the driver was previously observed; the other requirements remain unverified.
- Verify host CPU architecture, image architecture support, GPU compute capability support, and CUDA/PyTorch compatibility on the GB10. A detected GPU alone does not establish that the pinned image or segment CUDA extensions will run.
- Use GPU container access and `--shm-size=8g` for the documented Visual ChangeNet Docker path. Mount inputs and checkpoints with explicit container paths; provide writable results and extension-build storage.
- Check registry credential presence without exposing values. Use stdin for registry login. Any image pull, asset download, installation, or job launch needs the launch review/confirmation.

## Classify inference

1. Locate a trained classify checkpoint and its training configuration. A backbone-only `.safetensors` file is insufficient for defect inference. The skill warns that the old 7.0 NGC classify checkpoint fails against the newer C-RADIO architecture; do not download it as a substitute. Validate the supplied checkpoint against the selected runtime before bulk inference. If no compatible checkpoint exists, training is a separate scope requiring labeled training/validation data.
2. Match the backbone and classify head to that checkpoint. Default templates use C-RADIOv2-B; if its initialization asset is required, stage and mount it locally at `/data/pretrained_models/C-RADIOv2_B.safetensors`. DINOv3 checkpoints require their matching variant and the dedicated backbone instructions.
3. Supply a CSV with header `input_path,golden_path,label,object_name`. Input and golden paths are directories relative to the images root; `object_name` excludes extension and lighting suffix. The loader constructs paths such as `<images_dir>/<input_path>/<object_name>_SolderLight.jpg`, with the corresponding golden image under `golden_path`. Preserve the documented label column even for prediction; confirm accepted values for unlabeled data rather than inventing labels.
4. Set `dataset.classify.infer_dataset.images_dir` and `csv_path`, `image_ext`, `input_map`, `num_input`, dimensions and checkpoint-compatible preprocessing. Keep `model.classify.eval_margin` aligned with the validated decision threshold. Do not assume the template threshold is calibrated for this dataset.
5. Run `scripts/validate_vcn_dataset.py` with absolute host paths, `--mode inference`, the actual batch size/GPU count, and matching lighting/extension options. It must exit 0; fix failures before launch. Batch size must fit the sample count and memory; the skill specifies classify batch size >1, so a tiny sample set may need preparation before a smoke run.
6. Build a nested YAML spec from `references/spec_template_inference.yaml`, with `task: classify` and an explicit `inference.checkpoint`.

Action: `inference`. Container command: `visual_changenet inference -e <classify-spec.yaml>`.

Expected result: per-sample classify predictions and any supported scores/visualizations under the bound results directory. Inspect actual artifact names and score semantics during the smoke run; do not promise a particular CSV filename. Accuracy/FPR/FNR require a separate labeled evaluation run.

## Segment inference

1. Locate a trained segment checkpoint, distinct from the classify checkpoint. The skill lists an NGC LEVIR-CD segment model, but verify architecture/runtime compatibility and relevance to the target images before choosing or downloading it. A generic LEVIR change model does not establish PCB defect accuracy.
2. Match backbone and decoder settings to the checkpoint. The packaged segment template uses `vit_large_nvdinov2`; do not mix an NVDINOv2 checkpoint with a FAN backbone. A backbone initialization checkpoint alone is insufficient for a trained segmentation head.
3. Prepare same-named, spatially aligned before/after images under `A/` and `B/`. Put selected filenames in `list/test.txt`, or another explicitly selected prediction split. Set `dataset.segment.predict_split` to the split basename and `dataset.segment.root_dir` to the mounted root.
4. The documented dataset layout also includes `label/` containing binary mask PNGs (0 unchanged, 255 changed). Verify whether the selected runtime's prediction loader needs masks or only the directory. Do not fabricate ground-truth masks. Ground truth is necessary for quantitative evaluation, not for interpreting predicted masks as predictions.
5. Match `img_size`, class count, normalization, and multi-scale inference settings to the checkpoint. Check pair completeness, image readability, spatial alignment, split entries and available GPU memory. Choose a small smoke batch before scaling.
6. Verify the image can compile/load `MultiScaleDeformableAttention` on the target GPU. The skill estimates roughly five minutes for first-run CUDA-op compilation; reserve writable cache space and verify compiler/runtime compatibility.
7. Build a nested YAML spec from `references/spec_template_segment_inference.yaml`, with `task: segment` and an explicit `inference.checkpoint`.

Action: `segment_inference`. Container command: `visual_changenet inference -e <segment-spec.yaml>`. There is no separate `visual_changenet segment_inference` CLI subcommand.

Expected result: predicted change masks and supported visualizations in the segment results directory. Confirm mask resolution, class encoding, output count and correspondence with input pairs. IoU/F1 require a separate labeled evaluation run.

## Execution sequence once inputs are available

1. Inventory concrete checkpoint/data paths and original training specs. Select the execution platform and verify all paths from that compute environment.
2. Complete platform/model preflight and dataset checks. Resolve Docker permissions or choose another supported, reachable platform. Confirm image availability and compatibility before preparing runnable artifacts.
3. Present a launch review for both jobs: image, platform, checkpoints, datasets, GPU allocation, batches, output roots, configuration changes and expected runtime. Total inference time is unknown until sample counts and a smoke-run throughput measurement are available. Obtain confirmation before side effects.
4. Stage inputs only when storage locations require it. Write separate nested YAML specs; validate against the packaged action schemas. Disable optional external experiment tracking unless requested.
5. Open a separate job record and bind `results_dir` before each submission. Dispatch through the selected platform's `submit/status/logs/cancel` contract; no TAO SDK is needed.
6. Run a small representative smoke subset for each task. Verify checkpoint loading, data decoding, CUDA ops, finite outputs and expected artifacts before launching the full subsets.
7. Monitor backend state and logs, then record terminal states using `PENDING RUNNING COMPLETE ERROR CANCELED UNKNOWN`. Verify results and report artifact paths, processed counts, failures and limits of interpretation.

## Local references

- [Model skill](../tao-skill-bank/skills/models/tao-train-visual-changenet/SKILL.md)
- [Action/image contract](../tao-skill-bank/skills/models/tao-train-visual-changenet/references/skill_info.yaml)
- [Data formats](../tao-skill-bank/skills/models/tao-train-visual-changenet/references/data-formats.md)
- [Docker model instructions](../tao-skill-bank/skills/models/tao-train-visual-changenet/references/local-docker.md)
- [Docker platform](../tao-skill-bank/skills/platform/tao-run-on-docker/SKILL.md)
- [Launch gate](../tao-skill-bank/skills/core/tao-launch-workflow/SKILL.md)

`new-model/SKILL.md` was shown as an IDE tab but was not present on disk when this plan was prepared. `CurrentDirectory.md` was empty before this update.

Label QA update: the aligned pilot binary-label audit is complete: 134 accepted, 2 quarantined; all 180 occupancy records checked. Use `dataset-aligned-pilot/csv-reviewed/` for filtered pilot CSVs. See [labelQA.md](labelQA.md) for evidence and remaining production limits.

300-scene dataset pipeline started: seed 20261030; cycles full occupancy, valid top-to-bottom/leftward fill prefixes, and random gap layouts (30–60 vials). Track `dataset-300-run/status.json` and `dataset-300-run/render.log`. It automatically builds `dataset-300/` and runs TAO validators after rendering; outputs remain pending production QA. Raw images/placement manifests are in `dataset-300-renders/`. No training is launched.

Production QA update: all 14,971 pairs and 18,000 slots audited; no integrity errors or exact duplicates. A 144-crop stratified visual review quarantined 9 samples. Candidate CSVs retain 14,962 pairs, but release is on hold because random cap poses can invalidate source-derived defect labels and some views are obscured. A 34-pair real benchmark candidate was prepared. SLURM preflight stopped at SSH host-key verification. See [productionQA.md](productionQA.md) for results and remaining requirements.

Cap-label policy update: use relative cap/vial upward-axis tilt, below 1 degree PASS; at or above 1 degree, inverted or missing caps NO_PASS. Axial yaw does not count as tilt. The cutoff is geometric, not an established camera visibility threshold. Verified seed replay is updating all 300 sidecars without re-rendering images; automatic output is `dataset-300-tilt/`, tracked by `dataset-300-relabel/status.json`. Original dataset remains preserved. Obscured samples still require quarantine; this rule does not detect all possible non-tilt defects.

User accepted the relabeled dataset as sufficient to proceed. `dataset-300-tilt/acceptance.json` records checksums and retained limitations. Dataset: 14,971 occupied-vial pairs. Direct remote training replaces SLURM. Training preparation is blocked on runtime access: this session is on spark9, Docker socket access is denied, passwordless sudo is unavailable, and the active Python lacks PyTorch/TAO. No training was launched.

Native training environment prepared: `new-model/.envs/tao-vcn` (Python 3.12, CUDA-enabled PyTorch and source-based Visual ChangeNet). GPU operation and full training imports pass. Use direct execution on spark9; no SLURM/Docker required. See [training-environment.md](training-environment.md). Training is not launched.

Visual ChangeNet native GPU smoke test PASSED: one train optimizer step and one validation batch, finite losses, approximately 3.2 GB peak allocated GPU memory. See `smoke-test/result.json`, `smoke-test/run.log` and `smoke-test/spec.yaml`. Full training remains unstarted.

Full digital-dataset training launched and verified at optimizer step 1: `training-digital-001/`. Native GB10 run: 30 epochs, batch 8, AdamW lr 1e-5, validation every epoch, best val_loss and latest checkpoint retention. Augmentation remains disabled as in the verified smoke baseline. Track `training-digital-001/run-status.json`, `train.log`, and CSV metrics. No test-set data enters training.

Training completed all 30 epochs / 39,300 optimizer steps. A post-training completion-reporting callback selection error was fixed; no retraining required. Best checkpoint: `training-digital-001/model_best_028.pth` (epoch 29, validation loss 0.017311). `run-status.json` corrected to COMPLETE after verifying the saved model and all epoch metrics. Next: held-out synthetic and real-candidate evaluation.

Held-out synthetic evaluation COMPLETE: 2,285 pairs, threshold 0.5, accuracy 99.8687%, defect recall 98.9170%, TP 274 / TN 2008 / FP 0 / FN 3. Outputs: `evaluation-synthetic-001/evaluation-statistics.md`, `results.json`, `predictions.csv`, `failures.csv`. Threshold was not tuned on test data. Real-image evaluation remains pending.

Full-frame detection pipeline implemented with G01–G30 priority: `dataset-detection-G/` has 300 native full-frame images and scene-split COCO annotations (`vial`, `tilted_cap`, `missing_cap`) carrying slot IDs. Full-frame G-slot mapping and occupancy-gap postprocessor are available. No detector has been trained yet. See [FULL_FRAME_DETECTION.md](../csi_digital_twin_v1/tools/FULL_FRAME_DETECTION.md).

Order-completeness feature implemented: G01-G30 all occupied => complete; any empty G slot => incomplete. Output includes occupied count and empty slot IDs, independent of cap defects and fill-order gap errors. All 300 detection scenes updated (120 complete / 180 incomplete); unit and CLI integration checks pass.
