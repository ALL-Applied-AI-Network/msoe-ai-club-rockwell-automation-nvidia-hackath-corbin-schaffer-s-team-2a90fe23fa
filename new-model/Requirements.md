1. 
    HostName spark9.ai.msoe.edu
    User hackathon9

2. Classify checkpoint path: missing; training needed

3. Classify training YAML path and TAO version: no completed training run

4. GoldenImages/GoldenImage
new-model/capseal_review_batches_0 has the other crops


Native training environment prepared: `new-model/.envs/tao-vcn` (Python 3.12, CUDA-enabled PyTorch and source-based Visual ChangeNet). GPU operation and full training imports pass. Use direct execution on spark9; no SLURM/Docker required. See [training-environment.md](training-environment.md). Training is not launched.

Visual ChangeNet native GPU smoke test PASSED: one train optimizer step and one validation batch, finite losses, approximately 3.2 GB peak allocated GPU memory. See `smoke-test/result.json`, `smoke-test/run.log` and `smoke-test/spec.yaml`. Full training remains unstarted.

Full digital-dataset training launched and verified at optimizer step 1: `training-digital-001/`. Native GB10 run: 30 epochs, batch 8, AdamW lr 1e-5, validation every epoch, best val_loss and latest checkpoint retention. Augmentation remains disabled as in the verified smoke baseline. Track `training-digital-001/run-status.json`, `train.log`, and CSV metrics. No test-set data enters training.

Training completed all 30 epochs / 39,300 optimizer steps. A post-training completion-reporting callback selection error was fixed; no retraining required. Best checkpoint: `training-digital-001/model_best_028.pth` (epoch 29, validation loss 0.017311). `run-status.json` corrected to COMPLETE after verifying the saved model and all epoch metrics. Next: held-out synthetic and real-candidate evaluation.

Held-out synthetic evaluation COMPLETE: 2,285 pairs, threshold 0.5, accuracy 99.8687%, defect recall 98.9170%, TP 274 / TN 2008 / FP 0 / FN 3. Outputs: `evaluation-synthetic-001/evaluation-statistics.md`, `results.json`, `predictions.csv`, `failures.csv`. Threshold was not tuned on test data. Real-image evaluation remains pending.

Full-frame detection pipeline implemented with G01–G30 priority: `dataset-detection-G/` has 300 native full-frame images and scene-split COCO annotations (`vial`, `tilted_cap`, `missing_cap`) carrying slot IDs. Full-frame G-slot mapping and occupancy-gap postprocessor are available. No detector has been trained yet. See [FULL_FRAME_DETECTION.md](../csi_digital_twin_v1/tools/FULL_FRAME_DETECTION.md).

Order-completeness feature implemented: G01-G30 all occupied => complete; any empty G slot => incomplete. Output includes occupied count and empty slot IDs, independent of cap defects and fill-order gap errors. All 300 detection scenes updated (120 complete / 180 incomplete); unit and CLI integration checks pass.
