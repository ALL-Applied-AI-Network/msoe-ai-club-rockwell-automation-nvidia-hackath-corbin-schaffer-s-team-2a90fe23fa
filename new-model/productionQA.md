# 300-scene production QA

## Completed checks

- Checked 18,000 slot records across 300 scenes and both trays. Full and valid-prefix scenes have no gaps; random-gap scenes have 1,421 gaps, matching the specified fill order.
- Checked all 14,971 occupied-vial image pairs for readability, dimensions, golden-reference SHA-256, projected inspection-bound containment and source-name binary-label consistency. No integrity errors found.
- No exact JPEG duplicates, including across splits. This is not a perceptual near-duplicate or independent-source-family guarantee. Original vial source families overlap train/validation/test; these splits measure new randomized scenes, not new defect families.
- Visually inspected 144 stratified crops (12 per split/class/tray stratum). Review sheets and machine results are in `dataset-300/qa-production/`.
- Quarantined 9 sampled images for dark/obscured inspection regions or ambiguous label appearance; retained 14,962 candidate pairs. Filtered CSVs are in `dataset-300/csv-reviewed/` and pass TAO validators. Original files are preserved.

## Release remains on hold

A randomly rotated cap from source S01 can appear normally seated even though source identity still labels it NO_PASS. Thus source-family propagation is not sufficient to approve every post-randomization image. Some transparent uncapped vials against dark backgrounds are also not visually inspectable. Removing only the nine sampled examples does not establish that unsampled crops are safe.

Before training, either perform a full visibility/label review, or update generation to constrain capped defects to visibly defective poses and export geometric visibility/defect evidence, then regenerate and review affected samples. Preserve ambiguous samples as quarantine rather than changing labels based only on an uncertain crop. The full dataset has not been frozen as a training release; `candidate_checksums.json` freezes only the current candidate CSV/audit snapshot.

## Real-image benchmark candidate

Prepared `real-benchmark-candidate/test.csv` and paired images: 34 confirmed present-vial crops (28 PASS, 6 NO_PASS). Matched the real tray source to the annotation export's SHA-256 before cropping. Excluded draft/null/empty annotations and G24 itself. The six synthetic review batches are not real benchmark evidence.

This remains a candidate pending crop review. It comes from one frame, with the golden from the same frame, so it cannot establish performance across capture sessions. Provenance is saved in `real-benchmark-candidate/provenance.json`. The candidate passes TAO evaluation dataset validation.

## SLURM readiness

Read-only SSH preflight was attempted. Alias `inno-lab` did not resolve in this session. The explicit configured target `hackathon9@spark9.ai.msoe.edu` failed host-key verification. No trust settings were bypassed or changed. Establish/verify the host key through a trusted administrator or existing approved SSH configuration before retrying. Scheduler, GPU, container and shared-storage readiness remain unverified. No launch/training configuration was drafted against this unverified platform; no training job was submitted.
