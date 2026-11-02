# Pilot label QA

Reviewed all 136 occupied-vial crops across the three aligned pilot scenes, using the numbered sheets in `dataset-aligned-pilot/qa/label_audit_1.jpg` through `label_audit_3.jpg` and the fixed G24 golden. Source geometry was independently read from the tray USD.

**Result: 134 binary labels accepted; 2 samples quarantined.** No binary source-label propagation mismatches were found. Quarantined `capseal_0001_S30` and `capseal_0002_S25` have obscured inspection regions; retain their source NO_PASS labels in provenance, but exclude them from training/evaluation CSVs pending a clearer view. Do not relabel them PASS.

Use `dataset-aligned-pilot/csv-reviewed/` for the filtered pilot CSVs, with `dataset-aligned-pilot/images/` as the image root. Original CSVs/images remain unchanged for comparison. Exact decisions and reasons are in `dataset-aligned-pilot/qa/label_review.csv`; counts/checks are in `label_qa_summary.json` in the same directory.

The 35 source assemblies were audited for cap existence, authored rotations and root semantic attributes. No explicit good/bad class attributes were found on those roots. Consequently labels remain grounded in G/S source identity and the authored-tilt heuristic plus rendered visual review, not an exported semantic ground-truth attribute. No fine-grained defect subtype is approved by this audit. Transparent uncapped vials can look almost empty; use placement presence to distinguish them from empty slots.

All 180 slot records were checked (30 per tray per scene). The 40 gap flags agree with top-to-bottom then left fill order independently per tray: empty slots before the last occupied position are gaps; trailing empties are pending. These are spatial labels under the supplied rule, not evidence from an observed machine timeline. Empty slots remain excluded from present-vial quality CSVs.

PASS crops with neighboring caps in the background retain their target-vial label; background neighbors are not target defects. Blur, perspective and source-family overlap limit this pilot's usefulness as an independent performance benchmark. The audit approves the retained pilot's binary labeling, not training accuracy, defect visibility in future renders, or a production dataset release. Production generation must repeat visibility checks and include valid partial-fill prefixes as well as gaps.
