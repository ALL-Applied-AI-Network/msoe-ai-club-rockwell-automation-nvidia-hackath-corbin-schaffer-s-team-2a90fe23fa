# Vial dataset completion plan

## Decisions already supplied

- Golden reference: `new-model/GoldenImages/G24.jpg`, verified on disk as an 80 × 90 RGB image. Use this fixed image for every classify comparison. Preserve the original and record its SHA-256 in the dataset manifest.
- Generate inspection images from the tray-inspection digital twin using CapSealCam. Vials already have good/bad identity; carry that identity with each placed vial rather than relabeling rendered crops manually or inferring class from tray location.
- Start with binary classify training. `Requirements.md` says a classify checkpoint is missing and training is needed; its selected training platform is SLURM through SSH alias `inno-lab` (`hackathon9@spark9.ai.msoe.edu`). Rendering requires an Isaac Sim environment and is a separate execution prerequisite.
- Approved scope: begin a small pilot, then target 300 scenes (up to 18,000 occupied-vial crops) after pilot QA. Training is not launched.

## What exists and what is missing

`csi_digital_twin_v1/tools/render_example.py` already randomizes vial assemblies into distinct slots across two 30-slot trays, including copies of existing assemblies. It retains a per-record `broken` flag and pairs `chosen_records` with `chosen_slots`. Its current flag derives from S-series names or authored tilt; it is not a verified export of all semantic USD labels. It currently saves RGB images without a per-image placement/label manifest.

`generate_review_batches.py` already renders, prepares slot calibration, creates review pages and crops. The processing chain is a lens-corrected 640 × 480 image, upscale to 1920 × 1440, then bottom-right crop to 1920 × 1080. Crop coordinates must refer to the final image.

The provided real-image label export has 60 slots: 31 `pass`, 4 `missing_cap`, 3 `crooked_cap`, 1 `damaged_vial`, 20 `no_vial`, and 1 null class; 54 slots are confirmed and 6 draft. G24 is confirmed `pass` with presence true. These annotations belong to `tray_000001.jpg`; they must not be reused as labels for newly randomized scenes. Existing group summaries disagree with some slot-level occupancy fields, so use reviewed slot-level records rather than those summaries.

## Remaining work, in order

1. **Verify golden suitability and freeze preprocessing.** Visually check that G24 shows a complete, correctly capped vial and the relevant inspection area. Confirm the synthetic crops can match its framing, orientation and scale. Record crop padding, RGB handling, resize/letterbox policy, normalization and interpolation. Do not stretch arbitrary crops to 80 × 90 merely to match the golden; normalize both inputs consistently at the model's chosen input size. The training input size is still to be selected. Test golden-to-synthetic alignment on both tray locations; one real crop may need registration or restricted camera/view variation. Keep G24 fixed in all splits and exclude it as an inspection test example.

2. **Export a stable vial-to-slot manifest at render time.** Extend the renderer so every saved frame has a JSON sidecar containing image ID/path, seed, variation, camera/calibration and scene version, source assembly ID, placed instance/prim ID, slot ID, world transform, presence, semantic good/bad label, defect subtype when known and final-image crop box. Export all 60 slots including empty ones. Write the sidecar only for a successfully saved frame. Clear transform caches before projection. Preserve the mapping for cloned assemblies and retain source identity so copies can be grouped for splitting.

3. **Verify the label mapping survives randomization.** Resolve authoritative authored good/bad metadata or a reviewed source-assembly table, then propagate it to placed instances. Check the current `broken` heuristic against that source. Placement tray is not the label: a good vial placed in the scrap tray remains PASS. Review examples after pose/cap changes because the renderer straightens tilted vials and independently rotates/lifts caps on broken examples. Ensure NO_PASS defects remain visibly defective and PASS examples remain valid. Preserve subtype provenance; do not invent subtype labels from a binary flag.

4. **Apply the confirmed fill-order policy.** Within each tray, fill top-to-bottom, then move one column left. An empty position before the last occupied position is a gap error; trailing empty positions are awaiting refill. Record `present` and `gap_error` separately from vial PASS/NO_PASS labels. Export empty slots in the occupancy manifest, but exclude them from the present-vial quality CSV. Review occluded/truncated crops separately. Current gaps are spatial indicators under the supplied rule, not temporal proof of a machine failure.

5. **Crop by the actual placed slot.** Reuse the CapSeal calibration/export tools to project each occupied slot into final-image coordinates and extract an inspection crop with consistent padding. Account for lens distortion, upscale and the bottom-right crop. Carry each placed instance's label into its crop. Validate bounds, cap inclusion, neighboring-vial intrusion and all tray rows/columns using overlays. If camera geometry changes, regenerate calibration; static real-photo boxes from the label JSON cannot be reused for synthetic frames.

6. **Choose generation volume and variation coverage.** Set total frames/crops, class balance and per-subtype targets before the full render. Start with a small pilot spanning both trays, PASS and NO_PASS, full/partial occupancy and representative slots. Select deterministic nonoverlapping seed sets. Vary appearance only within plausible operating conditions (lighting/material/exposure and supported poses); keep camera geometry stable initially for the fixed golden. The current tool randomizes placement/count/poses, not a complete appearance variation pipeline. Set minimum accepted crop counts per split after pilot inspection; do not equate thousands of crops of a few copied assemblies with thousands of independent defect examples.

7. **Split scenes before augmentation or crop expansion.** Create reproducible train/validation/test assignments grouped by source scene/render sequence, so all crops from one frame stay together and near-duplicate frames stay together. Track source assembly families across clones and reserve distinct families/defect configurations where possible. If there are too few independent families, state that limitation. A proposed starting ratio is 70/15/15, subject to adequate independent PASS/NO_PASS samples per split. Freeze the test split and keep augmentation descendants in their parent's split. Shared G24 reference is intentional; inspection-image overlap is not.

8. **Export the TAO classify layout and CSVs.** Use one synthetic input channel with the loader's `SolderLight` filename suffix; this is a naming convention, not a claim that the rendered lighting reproduces solder illumination. For sample `frame0001_slotG01`, write an inspection image and a byte-identical copy of G24 named `frame0001_slotG01_SolderLight.jpg` in separate inspection/golden directories. CSV rows are `inspection,golden,PASS,frame0001_slotG01` or the verified defect label accepted by the binary loader. Keep exactly `input_path,golden_path,label,object_name` as the header, relative directory paths and unique stems. Verify PASS/defect mapping against the validator/runtime before bulk export. Emit train, validation and held-out test CSVs; an inference CSV can enumerate the held-out samples. Set `num_input: 1`, `input_map: {SolderLight: 0}`, `.jpg`, dimensions and preprocessing consistently in the later training spec. Choose the golden-copy strategy compatible with TAO's shared filename-stem contract; pointing every row directly at `G24.jpg` does not satisfy that loader convention.

9. **Keep a real-image benchmark.** Review available real crops and exclude unresolved draft/null annotations. Use reviewed real PASS/defect samples as an independent benchmark to measure the synthetic-to-real gap. Track original tray image/session as the group, not individual crop names. The supplied label JSON describes one real tray image, which cannot establish broad independent real-world performance. Locate its source crops, verify paths and G24 provenance, and collect additional independent real captures if needed. Never move benchmark examples into training after inspecting results without redefining the benchmark.

10. **Validate and release the dataset.** Check every CSV row resolves to both readable images, unique sample IDs, no inspection duplicates across splits, correct golden hashes and consistent dimensions. Run `validate_vcn_dataset.py` in `train` mode for train data and `evaluate`/`inference` modes for the respective CSVs, with actual paths, batch sizes, GPU counts, suffix and extension. Require two training classes and enough rows for each batch. Visually inspect contact sheets/overlays per class, split and slot. Save counts, exclusions, hashes, seeds, source manifest, split policy, preprocessing and dataset version. Record absolute local paths and eventual SLURM-visible staging paths; training cannot start until the dataset is readable from the compute environment.

## Proposed output layout

```text
new-model/dataset/                 # proposed; not created yet
  manifest.jsonl                  # source/instance/slot/label/crop/split provenance
  metadata.json                   # golden hash, preprocessing, seeds, versions
  images/
    inspection/<sample>_SolderLight.jpg
    golden/<sample>_SolderLight.jpg
  csv/train.csv
  csv/validation.csv
  csv/test.csv
  csv/inference.csv
  splits.json
  qa/                             # overlays, contact sheets, counts, exclusions
```

## Segment dataset: additional requirement

Good/bad instance labels and positions are enough for classify crops, but do not provide pixel-level defect masks for segment training. Segment data remains unfinished. It needs spatially aligned normal/defective image pairs and true change masks, generated with controlled paired renders/defect-region annotations or reviewed manually. Whole-vial instance masks and slot bounding boxes are not defect masks. G24 paired with arbitrary synthetic crops cannot be treated as pixel-aligned ground truth. If segment training remains required, design this paired-render/mask export separately and validate mask values (0 unchanged, 255 changed) before building `A/`, `B/`, `label/` and split lists.

## Completion criteria and checklist mapping

In `CurrentDirectory.md`, `[-]` means this dataset plan addresses the item but implementation/validation is pending; it does not mean the dataset is complete.

- Item 4: fulfilled when normalized inspection crops and corresponding G24 copies exist.
- Item 5: fulfilled when the valid held-out inference CSV exists; training/validation/test CSVs are additional outputs required by the new training scope.
- Item 6: partially fulfilled when dataset preprocessing is frozen; training architecture/configuration and a calibrated threshold still require training/validation.
- Item 10: partially fulfilled by selecting representative dataset smoke samples; results destinations remain separate.
- Items 7–9 remain open for segment-specific work. Dataset generation does not create checkpoints or solve platform access/credentials.

## Implementation and pilot status

The proposal has been initiated:

- `csi_digital_twin_v1/tools/render_example.py --export-placements` now writes a sidecar for each successfully rendered frame, carrying source/instance identity, slot order, position, good/bad label and occupancy gaps. Labels currently retain the existing source-name/authored-tilt rule; source-label and visual QA remain required.
- `new-model/tools/build_dataset.py` converts native placement-tagged renders to crops paired with byte-identical G24 copies, scene-grouped 70/15/15 CSV splits, provenance/occupancy records, metadata, overlays and a contact sheet. It refuses to overwrite an output directory. Empty slots are recorded, not fed to the vial-quality classifier. Source families may overlap splits and this limitation is reported.
- Verified syntax, six fill-order edge cases, and an isolated three-scene fixture export. The packaged TAO validators passed train, evaluate and inference modes with batch size 2. Fixture images were temporary test data and were not retained as training samples.
- Attempted the approved three-scene pilot (35–60 occupied slots, seed 20261029). Isaac Sim loaded the scene and randomized the vials, but produced no images because Vulkan/RTX device creation failed. Retrying with the installed NVIDIA ICD explicitly selected also failed with `ERROR_INCOMPATIBLE_DRIVER` / `vkCreateInstance failed`. CUDA detects GB10, but the graphics runtime is not operational in this session. This does not establish that a driver upgrade is the correct fix; driver/library/graphics-device access needs diagnosis first.
- Pilot logs: `/tmp/vial-dataset-pilot.log` and `/tmp/vial-dataset-pilot-nvidia.log`. No usable pilot images, crops or production dataset have been generated. Do not mark checklist items complete.

Next: restore a working NVIDIA Vulkan/RTX runtime on the render host, or use another Isaac Sim-capable render host; rerun the three-scene pilot, inspect crop/label/golden alignment, then generate 300 scenes. Prefer 210/45/45 scene assignments for the full dataset. Random occupancy in the pilot exercises gap errors; production coverage should also include valid partially filled prefixes and fully occupied trays rather than only random sparse layouts.

Once pilot renders exist, conversion is:

```bash
python new-model/tools/build_dataset.py \
  --renders new-model/dataset-pilot-renders \
  --golden new-model/GoldenImages/G24.jpg \
  --output new-model/dataset-pilot
```

The builder expects original 640 × 480 renderer outputs; do not pass images already enlarged by the review-batch tool. The full 300-scene run remains pending pilot QA and renderer recovery. Segment mask generation remains separate.

## Headless rendering recovered

The digital twin already uses headless `SimulationApp`; no display window is needed. This host has the Vulkan loader and NVIDIA graphics libraries but no `vulkaninfo` executable on PATH. The default NVIDIA ICD points to GLX, whose Vulkan instance entry point returned null here. A temporary ICD pointing to the installed NVIDIA EGL library successfully rendered the three-scene pilot. No driver installation or system configuration change was needed.

Reusable local launcher: `csi_digital_twin_v1/tools/isaac_headless.py`. It selects the verified EGL ICD and removes conflicting library overrides for the child process only. This is specific to the verified local install, not a portable driver replacement.

Pilot conversion produced **136 occupied-vial pairs**: train 49 (41 PASS/8 NO_PASS), validation 49 (42/7), test 38 (31/7). The manifest also records 40 gap errors across scenes. Outputs are in `new-model/dataset-pilot/`; raw scenes are in `new-model/dataset-pilot-renders/`. These pilot outputs supersede the earlier rendering-blocked status above.

Visual contact-sheet inspection found inconsistent cap centering, cut-off caps and neighboring-vial intrusion in several crops. The current slot-center crop needs adjustment to cap/inspection geometry before production generation; pilot data is not released for training. See `dataset-pilot/qa/contact_sheet.jpg` and overlays.

The user's successful existing image-generation command (run from `csi_digital_twin_v1`) was:

```bash
python3 tools/generate_review_batches.py --batches 6 --images-per-batch 100 \
  --min-vials 30 --max-vials 60 --output ../capseal_review_batches_0 --seed 42
```

That creates 600 review images using the same renderer. When needed on this session, select the EGL launcher by adding `--isaac-python tools/isaac_headless.py`. Existing output directories are intentionally rejected by the batch generator. Previously generated images do not automatically contain the new placement sidecars; do not infer their randomized vial identities from tray locations.

## Crop alignment corrected

The earlier pilot projected the vial base transform, which displaced the cap under perspective. The renderer now exports bounds for the vial mouth and cap geometry after randomization. The builder centers a crop on their union, adds padding, preserves its aspect ratio, and outputs 90 × 108 inspection images. Placement/gap coordinates remain separate from inspection geometry. Old sidecars without inspection bounds must be regenerated rather than silently reused.

Rerendered the same three seeded scenes into `new-model/dataset-aligned-renders/` and rebuilt `new-model/dataset-aligned-pilot/`: 136 occupied-vial pairs. Verified every retained crop contains its projected mouth/cap bounds, visually inspected the contact sheet, and passed packaged TAO train/evaluate/inference dataset validators at batch size 2. Caps are now centered and the previous systematic edge clipping is corrected. Physical occlusion, perspective/background differences and adjacent objects still present in the rendered scene are not removed by cropping. Source-label QA, real-golden compatibility and production dataset validation remain pending.

Use `dataset-aligned-pilot/qa/contact_sheet.jpg` and its per-scene overlays for current review; `dataset-pilot/` remains the uncorrected comparison. Production generation should use the updated renderer and builder.

Label QA update: the aligned pilot binary-label audit is complete: 134 accepted, 2 quarantined; all 180 occupancy records checked. Use `dataset-aligned-pilot/csv-reviewed/` for filtered pilot CSVs. See [labelQA.md](labelQA.md) for evidence and remaining production limits.

300-scene dataset pipeline started: seed 20261030; cycles full occupancy, valid top-to-bottom/leftward fill prefixes, and random gap layouts (30–60 vials). Track `dataset-300-run/status.json` and `dataset-300-run/render.log`. It automatically builds `dataset-300/` and runs TAO validators after rendering; outputs remain pending production QA. Raw images/placement manifests are in `dataset-300-renders/`. No training is launched.

Production QA update: all 14,971 pairs and 18,000 slots audited; no integrity errors or exact duplicates. A 144-crop stratified visual review quarantined 9 samples. Candidate CSVs retain 14,962 pairs, but release is on hold because random cap poses can invalidate source-derived defect labels and some views are obscured. A 34-pair real benchmark candidate was prepared. SLURM preflight stopped at SSH host-key verification. See [productionQA.md](productionQA.md) for results and remaining requirements.

Cap-label policy update: use relative cap/vial upward-axis tilt, below 1 degree PASS; at or above 1 degree, inverted or missing caps NO_PASS. Axial yaw does not count as tilt. The cutoff is geometric, not an established camera visibility threshold. Verified seed replay is updating all 300 sidecars without re-rendering images; automatic output is `dataset-300-tilt/`, tracked by `dataset-300-relabel/status.json`. Original dataset remains preserved. Obscured samples still require quarantine; this rule does not detect all possible non-tilt defects.

User accepted the relabeled dataset as sufficient to proceed. `dataset-300-tilt/acceptance.json` records checksums and retained limitations. Dataset: 14,971 occupied-vial pairs. Direct remote training replaces SLURM. Training preparation is blocked on runtime access: this session is on spark9, Docker socket access is denied, passwordless sudo is unavailable, and the active Python lacks PyTorch/TAO. No training was launched.

Full-frame detection pipeline implemented with G01–G30 priority: `dataset-detection-G/` has 300 native full-frame images and scene-split COCO annotations (`vial`, `tilted_cap`, `missing_cap`) carrying slot IDs. Full-frame G-slot mapping and occupancy-gap postprocessor are available. No detector has been trained yet. See [FULL_FRAME_DETECTION.md](../csi_digital_twin_v1/tools/FULL_FRAME_DETECTION.md).
