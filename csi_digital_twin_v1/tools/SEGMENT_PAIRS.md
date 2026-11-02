# Controlled segmentation pairs

From the repository root:

```bash
python3 csi_digital_twin_v1/tools/isaac_headless.py \
  csi_digital_twin_v1/tools/render_example.py \
  --camera CapSealCam \
  --stage stations/tray_inspection/tray_inspection.usdc \
  --segment-pairs new-model/dataset-segment-pilot \
  --variations 10 --min-vials 35 --max-vials 60 --seed 42 \
  --segment-defect tilt --segment-tilt-degrees 15 \
  --segment-unchanged-every 5
```

Random defects with 300 data points
```bash
python3 csi_digital_twin_v1/tools/isaac_headless.py \
csi_digital_twin_v1/tools/render_example.py   \
--camera CapSealCam   \
--stage stations/tray_inspection/tray_inspection.usdc   \
--segment-pairs new-model/dataset-segment-pilot   \
--variations 300 \
--min-vials 35 \
--max-vials 60 \
--seed 16   \
--segment-defect random \
--segment-tilt-degrees 25   \
--segment-unchanged-every 20
```

Use a new output directory. Pair mode implies tray randomization, chooses normal
capped source assemblies (cloning them if necessary), and renders A before
modifying one selected cap for B. Camera, lights, placements, timeline, and lens
processing stay fixed within each pair. The cap state is restored afterward.
`--variations` counts pairs. `--out` is unused in this mode.

- `--segment-defect tilt`: rotate one cap about its authored transform origin.
  The tilt is synthetic and may intersect the vial; visually review suitability.
- `--segment-defect missing`: hide one cap in B.
- `--segment-defect none`: normal/normal pairs only.
- `--segment-unchanged-every N`: every Nth pair is normal/normal; 0 disables this.

Outputs: matching full-frame PNGs in `A/` and `B/`, `list/test.txt`, and
`scenes/pair_*.json` plus flattened normal-scene USD snapshots. Metadata records
the seed, camera calibration, transforms, target cap, defect, and render settings.
USD snapshots can still reference external assets; preserve the source assets.

No independent crop or resize is applied: both images retain the same full-frame
coordinates (currently 640 x 480 for CapSealCam). If extracting vial crops later,
use one shared box encompassing normal and defective geometry for both images.
Do not recenter each image separately. Check occlusion and defect visibility.
Normal/normal renders may have sampling noise; do not infer masks by subtraction.

This exports inference pairs, not a released training dataset. No pixel masks are
created. Training/evaluation needs reviewed ground-truth masks and scene-grouped
train/validation/test splits. `list/test.txt` lists generated pairs only and does
not establish an independent held-out test set.
