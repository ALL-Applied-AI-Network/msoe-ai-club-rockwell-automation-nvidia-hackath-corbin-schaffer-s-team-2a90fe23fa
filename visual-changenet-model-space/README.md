# Visual ChangeNet model space: G09

This workspace follows the `tao-train-visual-changenet` classification data
contract. The supplied `G09.jpg` crop is staged as both the observed and golden
image, making it a single PASS pair suitable for inference smoke testing.

It is not a training dataset. Visual ChangeNet classification training requires
both PASS and non-PASS pairs, and a fresh TAO 7.3-compatible classification
checkpoint must be trained from the C-RADIOv2-B backbone before inference.

## Layout

```text
datasets/inference.csv
images/observed/G09_SolderLight.jpg
images/golden/G09_SolderLight.jpg
models/README.md
results/README.md
specs/inference.yaml
```

TAO resolves each CSV row as:

```text
<images_dir>/<input_path>/<object_name>_SolderLight.jpg
<images_dir>/<golden_path>/<object_name>_SolderLight.jpg
```

## Validate the staged pair

From the repository root:

```bash
python3 tao-skill-bank/skills/models/tao-train-visual-changenet/scripts/validate_vcn_dataset.py \
  --csv visual-changenet-model-space/datasets/inference.csv \
  --images-dir visual-changenet-model-space/images \
  --mode inference \
  --batch-size 1 \
  --num-gpus 1
```

## Run inference later

1. Put a TAO 7.3 Visual ChangeNet classification checkpoint at
   `models/changenet_model_classify_latest.pth`.
2. Ensure Docker, NVIDIA Container Toolkit, and an authenticated `nvcr.io`
   session are available.
3. Mount this directory at `/workspace` and run the pinned image:

```bash
docker run --rm --gpus all --shm-size=8g \
  -v "$PWD/visual-changenet-model-space:/workspace" \
  nvcr.io/nvidia/tao/tao-toolkit:7.3.0-pyt \
  visual_changenet inference -e /workspace/specs/inference.yaml
```

The spec uses nested TAO configuration keys and writes output beneath
`results/inference/`.
