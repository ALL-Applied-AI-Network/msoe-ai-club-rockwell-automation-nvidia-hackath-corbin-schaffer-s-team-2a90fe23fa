# Native training environment

Created an isolated conda environment at `new-model/.envs/tao-vcn` with Python 3.12, leaving `hackathon9` unchanged. The existing `hackathon9` environment uses Python 3.15 and has no PyTorch/TAO dependencies.

Verified CUDA-enabled PyTorch 2.11.0 + CUDA 13.0, torchvision 0.26.0, GB10 detection, and an actual GPU matrix operation. Installed the Visual ChangeNet native dependencies, including NVIDIA EFF and compiled PyCUDA. Full Visual ChangeNet training-module imports pass.

Official TAO source is pinned locally to commit `99403b7dd941d1276e4e23ed3efd6794e207a75d` in `new-model/vendor/tao-pytorch`; a `.pth` file in the isolated environment exposes that source. This is a source-based native runtime, not the old TAO 5.1 package from NVIDIA's pip index and not the container runtime. Dependency versions are recorded in `training-environment.lock.txt`. Environment and vendor directories are excluded from Git.

From the repository root:

```bash
conda activate ./new-model/.envs/tao-vcn
export PYTHONNOUSERSITE=1
python -m nvidia_tao_pytorch.cv.visual_changenet.scripts.train --help
```

Alternatively, invoke `new-model/.envs/tao-vcn/bin/python -s` directly to avoid user-site package contamination. No Docker, SLURM, administrator permissions or system package changes are used.

Environment import/GPU checks do not replace the pending dataset/backbone/configuration smoke test. Training has not been launched. The native source schema must be checked against the final training spec before launch; do not assume it is identical to the skill bank's container schema.

Packaging caveat: `pip check` reports `nvidia-cusparselt-cu13 0.8.0 is not supported on this platform`. Its installed wheel declares `manylinux2014_sbsa` while the machine reports `aarch64`; it was pulled by the official CUDA PyTorch dependency chain. GPU tensor operations and Visual ChangeNet imports/help succeeded, but sparse-library/model execution has not been verified. Do not treat this environment as fully validated until the model smoke test passes.

## Model smoke test passed

`smoke-test/result.json` records one completed GPU optimizer step and one validation batch using the accepted `dataset-300-tilt` and existing `base-model/model.safetensors`. Train loss 0.58429, validation loss 0.68169, peak allocated GPU memory 3,190,637,568 bytes. These are execution checks, not accuracy estimates. No full training or checkpoint save was performed.

Built the required `MultiScaleDeformableAttention` operator from the pinned TAO source for CUDA architecture 12.1 using `tools/build_vcn_ops.py`. The validated spec explicitly sets `.jpg`, decoder feature strides `[4,8,16,32]`, C-RADIO summary token enabled, CLIP normalization (via the official data module) and one SolderLight input. The source schema's defaults alone were insufficient. The resolved smoke specification is `smoke-test/spec.yaml`; reproducible script is `tools/smoke_vcn.py`.

The earlier cuSPARSELt platform-tag warning remains a packaging caveat, but the actual Visual ChangeNet dense forward/backward/optimizer and validation paths have now executed successfully on GB10. Sparse operations were not specifically exercised.
