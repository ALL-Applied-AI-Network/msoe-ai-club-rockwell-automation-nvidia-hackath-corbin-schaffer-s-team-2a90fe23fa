#!/usr/bin/env python3
"""Export the trained detector to a Windows-friendly ONNX CPU bundle."""
import argparse
import hashlib
import json
import shutil
import sysconfig
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf
from nvidia_tao_pytorch.cv.rtdetr.model.pl_rtdetr_model import RTDETRPlModel


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "final_model/exports/windows-x64-cpu"
PACKAGE_FILES = (
    "README.md",
    "requirements.txt",
    "run.py",
    "runtime/full_frame_detection.py",
    "runtime/temporal.py",
    "calibration.json",
)


def bundle_files() -> list[Path]:
    return sorted(
        path
        for path in BUNDLE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )


class WindowsCPUDetector(nn.Module):
    """Fixed-shape native-frame detector with a stable 300-row output."""

    def __init__(self, detector: nn.Module, eval_size: tuple[int, int], num_select: int):
        super().__init__()
        self.detector = detector
        self.eval_size = eval_size
        self.num_select = num_select
        self.antialias = True

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(
            image,
            size=self.eval_size,
            mode="bilinear",
            align_corners=False,
            antialias=self.antialias,
        )
        image = torch.round(image.clamp(0, 1) * 255.0) / 255.0
        outputs = self.detector(image)
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        scores, indices = torch.topk(logits.sigmoid().flatten(1), self.num_select, dim=1)
        labels = indices % logits.shape[-1]
        queries = torch.div(indices, logits.shape[-1], rounding_mode="floor")
        xyxy = torch.cat(
            (boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2),
            dim=-1,
        )
        xyxy = torch.gather(xyxy, 1, queries.unsqueeze(-1).expand(-1, -1, 4))
        scale = xyxy.new_tensor([640, 480, 640, 480])
        xyxy = xyxy * scale
        return torch.cat(
            (xyxy, scores.unsqueeze(-1), labels.to(xyxy.dtype).unsqueeze(-1)),
            dim=-1,
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enable_input_resize_antialias(model: onnx.ModelProto) -> None:
    """Set ONNX Resize's opset-18 antialias attribute for the detector input only."""
    resized_inputs = [
        node for node in model.graph.node if node.op_type == "Resize" and "image" in node.input
    ]
    if len(resized_inputs) != 1:
        raise RuntimeError(f"Expected one input Resize node, found {len(resized_inputs)}.")
    node = resized_inputs[0]
    for attribute in list(node.attribute):
        if attribute.name == "antialias":
            node.attribute.remove(attribute)
    node.attribute.append(onnx.helper.make_attribute("antialias", 1))


def build_model(checkpoint_path: Path, spec_path: Path) -> tuple[nn.Module, OmegaConf]:
    cfg = OmegaConf.load(spec_path)
    if cfg.dataset.augmentation.preserve_aspect_ratio:
        raise ValueError("This exporter only supports the trained direct-resize pipeline.")
    if cfg.dataset.remap_mscoco_category:
        raise ValueError("Expected the trained custom categories 1, 2, and 3.")
    cfg.model.pretrained_backbone_path = ""
    model = RTDETRPlModel(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    detector = model.model.cpu().eval()
    adapter = WindowsCPUDetector(
        detector,
        tuple(cfg.dataset.augmentation.eval_spatial_size),
        int(cfg.model.num_select),
    ).eval()
    return adapter, cfg


def confidence_filtered(rows: np.ndarray, threshold: float) -> np.ndarray:
    return rows[
        (rows[:, 5] >= 1)
        & (rows[:, 5] <= 3)
        & (rows[:, 4] >= threshold)
    ]


def compare_detection_sets(
    expected: np.ndarray,
    actual: np.ndarray,
    description: str,
) -> dict[str, float | int]:
    if len(expected) != len(actual):
        raise AssertionError(
            f"{description}: detection counts differ ({len(expected)} != {len(actual)})."
        )
    minimum_iou = 1.0
    maximum_score_delta = 0.0
    for category in (1, 2, 3):
        left = expected[expected[:, 5] == category]
        right = actual[actual[:, 5] == category]
        if len(left) != len(right):
            raise AssertionError(
                f"{description}: category {category} counts differ ({len(left)} != {len(right)})."
            )
        if not len(left):
            continue
        top_left = np.maximum(left[:, None, :2], right[None, :, :2])
        bottom_right = np.minimum(left[:, None, 2:4], right[None, :, 2:4])
        intersection = np.prod(np.maximum(0, bottom_right - top_left), axis=2)
        left_area = np.prod(np.maximum(0, left[:, 2:4] - left[:, :2]), axis=1)
        right_area = np.prod(np.maximum(0, right[:, 2:4] - right[:, :2]), axis=1)
        ious = intersection / np.maximum(
            left_area[:, None] + right_area[None, :] - intersection,
            np.finfo(np.float32).eps,
        )
        remaining_left = list(range(len(left)))
        remaining_right = list(range(len(right)))
        while remaining_left:
            local = ious[np.ix_(remaining_left, remaining_right)]
            local_left, local_right = np.unravel_index(np.argmax(local), local.shape)
            left_index = remaining_left.pop(local_left)
            right_index = remaining_right.pop(local_right)
            overlap = float(ious[left_index, right_index])
            score_delta = float(abs(left[left_index, 4] - right[right_index, 4]))
            if overlap < 0.95:
                raise AssertionError(
                    f"{description}: category {category} box IoU {overlap:.5f} is below 0.95."
                )
            if score_delta > 0.03:
                raise AssertionError(
                    f"{description}: category {category} score delta {score_delta:.5f} exceeds 0.03."
                )
            minimum_iou = min(minimum_iou, overlap)
            maximum_score_delta = max(maximum_score_delta, score_delta)
    return {
        "detections": len(expected),
        "minimum_iou": minimum_iou,
        "maximum_score_difference": maximum_score_delta,
    }


def validate_frames(
    onnx_path: Path,
    reference_model_path: Path,
    reference_operator_path: Path,
    dataset_path: Path,
    cpu_model: nn.Module,
) -> list[dict[str, float | int | str]]:
    if not reference_model_path.is_file() or not reference_operator_path.is_file():
        raise FileNotFoundError("The existing GB10 TorchScript model and operator are required for parity checks.")
    import nvidia_tao_pytorch

    loaded_operator = (
        Path(nvidia_tao_pytorch.__file__).parent
        / "cv/deformable_detr/model/ops"
        / ("MultiScaleDeformableAttention" + sysconfig.get_config_var("EXT_SUFFIX"))
    )
    if sha256(loaded_operator) != sha256(reference_operator_path):
        raise ValueError("The operator loaded by TAO does not match the GB10 export operator.")
    reference_metadata_path = reference_model_path.with_suffix(".json")
    reference_metadata = json.loads(reference_metadata_path.read_text())
    expected_hashes = {
        item["file"]: item["sha256"]
        for item in reference_metadata.get("required_libraries", [])
    }
    if expected_hashes.get(reference_operator_path.name) != sha256(reference_operator_path):
        raise ValueError("The GB10 model metadata does not match its inference operator.")
    reference = torch.jit.load(str(reference_model_path), map_location="cuda").eval()
    images = json.loads((dataset_path / "annotations/validation-keyframes.json").read_text())["images"]
    indices = sorted(set((0, len(images) // 3, 2 * len(images) // 3, len(images) - 1)))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    results = []
    for index in indices:
        image_path = dataset_path / "frames" / images[index]["file_name"]
        with Image.open(image_path) as source:
            array = np.asarray(source.convert("RGB")).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float() / 255
        with torch.inference_mode():
            gb10_rows = reference(tensor.to("cuda")).cpu().numpy()
            cpu_rows = cpu_model(tensor).numpy()[0]
        onnx_rows = session.run(["detections"], {"image": tensor.numpy()})[0][0]
        cpu_detections = confidence_filtered(cpu_rows, 0.1)
        onnx_detections = confidence_filtered(onnx_rows, 0.1)
        gb10_detections = confidence_filtered(gb10_rows, 0.1)
        cpu_onnx = compare_detection_sets(cpu_detections, onnx_detections, f"frame {index}, CPU")
        gb10_onnx = compare_detection_sets(gb10_detections, onnx_detections, f"frame {index}, GB10")
        threshold_counts = {}
        for name, rows in (("torch_cpu", cpu_rows), ("onnx_cpu", onnx_rows), ("gb10", gb10_rows)):
            threshold_counts[name] = {
                str(category): int(
                    ((rows[:, 5] == category) & (rows[:, 4] >= 0.25)).sum()
                )
                for category in (1, 2, 3)
            }
        threshold_disagreements = [
            {
                "category_id": category,
                "counts": {
                    name: threshold_counts[name][str(category)]
                    for name in ("torch_cpu", "onnx_cpu", "gb10")
                },
            }
            for category in (1, 2, 3)
            if len(
                {
                    threshold_counts[name][str(category)]
                    for name in ("torch_cpu", "onnx_cpu", "gb10")
                }
            )
            > 1
        ]
        results.append(
            {
                "frame_index": index,
                "confidence_threshold_for_parity": 0.1,
                "cpu_onnx_parity": cpu_onnx,
                "gb10_onnx_parity": gb10_onnx,
                "counts_at_runtime_default_0_25": threshold_counts,
                "threshold_0_25_disagreements": threshold_disagreements,
            }
        )
    return results


def create_bundle(
    onnx_path: Path,
    checkpoint_path: Path,
    validation: list[dict[str, float | int | str]],
) -> Path:
    calibration_source = BUNDLE.parent / "rtdetr-v3-006-final/calibration.json"
    calibration_target = BUNDLE / "calibration.json"
    if not calibration_target.exists():
        shutil.copy2(calibration_source, calibration_target)
    for relative in PACKAGE_FILES:
        if not (BUNDLE / relative).is_file():
            raise FileNotFoundError(f"Missing Windows package file: {BUNDLE / relative}")
    model_path = BUNDLE / "model/rtdetr-v3-006-final.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if onnx_path.resolve() != model_path.resolve():
        shutil.copy2(onnx_path, model_path)
    metadata = {
        "format": "ONNX",
        "execution_provider": "CPUExecutionProvider",
        "input": {"name": "image", "shape": [1, 3, 480, 640], "dtype": "float32", "color": "RGB"},
        "output": {
            "name": "detections",
            "shape": [1, 300, 6],
            "columns": ["x0", "y0", "x1", "y1", "score", "category_id"],
            "category_zero": "unused; ignored by the runtime",
        },
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "parity_method": (
            "One-to-one, same-class box matching at confidence >=0.1; minimum IoU 0.95 "
            "and maximum score difference 0.03."
        ),
        "parity_check": validation,
        "runtime_threshold_note": (
            "CPU/GPU arithmetic can change detections close to the confidence threshold. "
            "See counts_at_runtime_default_0_25 and threshold_0_25_disagreements."
        ),
    }
    (BUNDLE / "model/model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    package_files = bundle_files()
    manifest = {
        path.relative_to(BUNDLE).as_posix(): sha256(path)
        for path in package_files
        if path.name != "manifest.json"
    }
    (BUNDLE / "manifest.json").write_text(json.dumps({"sha256": manifest}, indent=2) + "\n")
    archive_path = BUNDLE.parent / "rtdetr-v3-006-final-windows-x64-cpu.zip"
    if archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {archive_path}")
    temporary_archive = archive_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in bundle_files():
            archive.write(path, path.relative_to(BUNDLE.parent))
    temporary_archive.replace(archive_path)
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "final_model/training-v3-006/final_epoch_327.pth")
    parser.add_argument("--spec", type=Path, default=ROOT / "final_model/training-v3-006/spec.yaml")
    parser.add_argument("--reference-model", type=Path, default=ROOT / "final_model/models/rtdetr-v3-006-final.ts")
    parser.add_argument(
        "--reference-operator",
        type=Path,
        default=ROOT / "final_model/models/rtdetr-v3-006-final.ops.so",
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "final_model/dataset-v3")
    parser.add_argument("--output", type=Path, default=BUNDLE / "model/rtdetr-v3-006-final.onnx")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    for path in (args.checkpoint, args.spec, args.dataset, args.reference_model, args.reference_operator):
        if not path.exists():
            parser.error(f"Required input does not exist: {path}")
    if args.output.exists() or (BUNDLE.parent / "rtdetr-v3-006-final-windows-x64-cpu.zip").exists():
        parser.error("Windows export already exists; choose a fresh output location or remove it explicitly.")
    if not torch.cuda.is_available():
        parser.error("Export parity verification requires the existing GB10 CUDA runtime.")
    if torch.cuda.get_device_capability() != (12, 1):
        parser.error("Expected the verified GB10 CUDA device for the reference comparison.")
    model, _ = build_model(args.checkpoint, args.spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros((1, 3, 480, 640), dtype=torch.float32)
    export_path = args.output.with_name(args.output.stem + ".partial.onnx")
    if export_path.exists():
        parser.error(f"Temporary export already exists: {export_path}")
    try:
        model.antialias = False
        try:
            torch.onnx.export(
                model,
                (sample,),
                export_path,
                input_names=["image"],
                output_names=["detections"],
                opset_version=args.opset,
                dynamo=False,
                do_constant_folding=True,
                external_data=True,
            )
        finally:
            model.antialias = True
        onnx_model = onnx.load(str(export_path), load_external_data=True)
        enable_input_resize_antialias(onnx_model)
        onnx.save_model(onnx_model, str(export_path), save_as_external_data=False)
        onnx.checker.check_model(onnx_model)
        session = ort.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError(f"Unexpected ONNX Runtime providers: {session.get_providers()}")
        validation = validate_frames(
            export_path,
            args.reference_model,
            args.reference_operator,
            args.dataset,
            model,
        )
        export_path.replace(args.output)
    except BaseException:
        export_path.unlink(missing_ok=True)
        raise
    archive_path = create_bundle(args.output, args.checkpoint, validation)
    print(json.dumps({"archive": str(archive_path), "size_bytes": archive_path.stat().st_size, "parity": validation}, indent=2))


if __name__ == "__main__":
    main()
