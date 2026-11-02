# Full-frame detection, G tray first

The default inspection scope is **G01–G30**. Frames remain the original 640 × 480 camera images; no individual vial crops or golden input are required for the detector dataset. S-tray objects remain visible but are out of scope by default. Use `--scope both` / `--detection-scope both` only when both trays should be inspected.

The exporter copies the existing crop pipeline's slot IDs verbatim. It does not renumber slots based on source vial names. A source S vial placed at G12 is labeled G12. G positions run top-to-bottom, then one column left; S01–S30 are distinct IDs when enabled.

## Export from existing verified scenes

From the repository root, choose a new output directory:

```bash
python csi_digital_twin_v1/tools/export_detection_dataset.py \
  --placements new-model/dataset-300-relabel \
  --splits new-model/dataset-300-tilt/splits.json \
  --scope G --output new-model/dataset-detection-G
```

That export has already been created and validated: 210/45/45 full-frame scenes in train/validation/test. Each COCO annotation includes `slot_id`, `tray`, source identity, and tilt in addition to standard COCO fields. All images have slot metadata, including scenes without defects. Existing scene split assignments are preserved.

Categories:

1. `vial`: top inspection region of every present vial, enabling occupancy detection.
2. `tilted_cap`: cap tilt at least 1° relative to the vial (including inverted caps).
3. `missing_cap`: present vial without a cap.

The `vial` and defect boxes may overlap for the same slot. Boxes encompass projected mouth/cap geometry, not pixel-exact defect outlines. Occlusion QA remains necessary. Normal caps below 1° produce no defect annotation. Empty gaps are stored separately in slot metadata; they are not missing-cap annotations and are not trained as visually local defect boxes.

Outputs: `images/`, COCO `annotations/{train,validation,test}.json`, per-image `slots/`, G-slot `calibration.json`, labeled full-scene overlays in `qa/`, and count/validation summaries.

## Export annotations during future renders

Add `--export-detections --detection-scope G` to a randomized CapSeal render. This implies placement export and writes `.detections.json` next to each full image. It is a separate mode from segmentation pairs and cannot be used with `--placements-only`. Existing segmentation-pair behavior is preserved.

## Attach G IDs to detector predictions

A trained detector is still required; the old paired classifier is not a full-frame detector. Once full-frame inference produces COCO-style results (`image_id`, `bbox` in xywh, `category_id`, `score`), run:

```bash
python csi_digital_twin_v1/tools/label_full_frame_predictions.py \
  --predictions predictions.json \
  --calibration new-model/dataset-detection-G/calibration.json \
  --scope G --score-threshold 0.1 --output slot_predictions.json
```

The 0.1 cutoff is an initial candidate, not a calibrated operating point. Predicted boxes must first be mapped back from detector resizing/letterboxing to the original native image coordinates. The fixed-camera calibration maps to G slots using median normal-cap anchors; predictions farther than half a slot pitch are left unassigned, rather than forcing S-tray detections onto G slots. Recalibrate for a changed view or real camera. COCO slot attributes alone do not teach standard detectors to output slot identities.

The postprocessor includes `slot_id` on defects and separate `possible_gap_slots`, using the observed fill order on G. Vial detection misses can create false gap warnings; no visible vials means no inferred fill boundary. These are spatial occupancy findings, not evidence of the machine's future refill behavior.

## Verification and remaining work

`test_full_frame_detection.py` covers the 1° boundary, missing caps, G-only scope, empty gaps, box clipping and distant prediction rejection. Structural validation checked every exported annotation, full-frame image dimension, slot range and split image ID. Visual overlays were generated for each split. Detection training and real-camera calibration are not performed by this change. Next: review overlays/occlusion policy, configure and smoke-test RT-DETR, then train with these full-frame COCO files.

## G-tray order completeness

Every frame now carries `good_tray_order`:

```json
{
  "order_status": "incomplete",
  "order_complete": false,
  "occupied_count": 29,
  "required_count": 30,
  "empty_slots": ["G12"]
}
```

Only occupancy of G01–G30 determines this status: all 30 occupied means complete; any empty slot means incomplete. S-tray occupancy never counts, and duplicate detections do not fill additional slots. A full tray may still have cap defects. A partial tray can be incomplete without having a fill-order gap.

Synthetic annotations use actual placement presence. At inference the status is derived from confidence-filtered detections assigned to slots, so missed/occluded vial detections can cause an incomplete warning. This is implemented postprocessing, not a new image-classification head, and still needs a trained full-frame detector.

Pass `--images <COCO annotation file>` to `label_full_frame_predictions.py` so frames with zero predictions are included and marked incomplete (all 30 slots empty). Without an image manifest, only image IDs appearing in the predictions can be reported. Coordinates must match the calibration frame.

The existing 300-scene export has been updated: 120 complete and 180 incomplete G trays, recorded in `new-model/dataset-detection-G/order-summary.json`. Neither completeness nor gap status is a new COCO box class; both are derived from slot occupancy.
