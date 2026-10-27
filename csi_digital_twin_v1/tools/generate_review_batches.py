#!/usr/bin/env python3
"""Render CapSeal image batches and prepare one vial-labeler review per image."""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

TOOLS = Path(__file__).resolve().parent
TWIN = TOOLS.parent
REPO = TWIN.parent
ISAAC_PYTHON = REPO / "IsaacSim/_build/linux-aarch64/release/python.sh"
LABELER = REPO / "vial-labeler-participants-v1/labeler.py"
STAGE = TWIN / "stations/tray_inspection/tray_inspection.usdc"
CAMERA = TWIN / "cameras/CapSealCam.json"
TARGET_SIZE = (1920, 1080)


def upscale_bottom_right(image_path):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    target_width, target_height = TARGET_SIZE
    scale = max(target_width / image.width, target_height / image.height)
    resized_size = (round(image.width * scale), round(image.height * scale))
    image = image.resize(resized_size, Image.Resampling.LANCZOS)
    left = resized_size[0] - target_width
    top = resized_size[1] - target_height
    image = image.crop((left, top, left + target_width, top + target_height))
    temporary = image_path.with_name(image_path.stem + ".upscaled.png")
    image.save(temporary)
    temporary.replace(image_path)


def run(command, *, cwd=None):
    command = [str(part) for part in command]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=positive_int, required=True,
                        help="number of numbered batch folders to create")
    parser.add_argument("--images-per-batch", type=positive_int, required=True,
                        help="number of randomized images in each batch")
    parser.add_argument("--output", type=Path, required=True,
                        help="new output directory; an existing directory is never overwritten")
    parser.add_argument("--seed", type=int, default=42,
                        help="base randomization seed; each batch uses a distinct seed")
    parser.add_argument("--min-vials", type=int, default=60)
    parser.add_argument("--max-vials", type=int, default=60)
    parser.add_argument("--broken-lean-degrees", type=float, default=0.0)
    parser.add_argument("--isaac-python", type=Path, default=ISAAC_PYTHON)
    parser.add_argument("--labeler-python", default=sys.executable)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    isaac_python = args.isaac_python.expanduser().resolve()
    if output.exists():
        parser.error(f"output already exists: {output}; choose a new directory")
    if args.min_vials < 0 or args.max_vials < args.min_vials or args.max_vials > 60:
        parser.error("vial counts must satisfy 0 <= --min-vials <= --max-vials <= 60")
    if not 0 <= args.broken_lean_degrees <= 3:
        parser.error("--broken-lean-degrees must be between 0 and 3")
    if not isaac_python.is_file():
        parser.error(f"Isaac Sim Python was not found: {isaac_python}")
    if not LABELER.is_file():
        parser.error(f"vial-labeler was not found: {LABELER}")

    with tempfile.TemporaryDirectory(prefix="capseal-calibration-") as temp_dir:
        calibration = Path(temp_dir) / "capseal_calibration.json"
        run(
            [
                isaac_python,
                TOOLS / "export_labeler_calibration.py",
                "--stage", STAGE,
                "--camera-json", CAMERA,
                "--output", calibration,
            ],
            cwd=TWIN,
        )

        output.mkdir(parents=True)
        for batch_number in range(1, args.batches + 1):
            batch = output / f"{batch_number:03d}"
            images = batch / "images"
            html = batch / "html"
            images.mkdir(parents=True)
            html.mkdir()
            run(
                [
                    isaac_python,
                    TOOLS / "render_example.py",
                    "--camera", "CapSealCam",
                    "--stage", STAGE,
                    "--randomize-trays",
                    "--variations", args.images_per_batch,
                    "--min-vials", args.min_vials,
                    "--max-vials", args.max_vials,
                    "--broken-lean-degrees", args.broken_lean_degrees,
                    "--seed", args.seed + batch_number - 1,
                    "--out", images / "cap_seal.png",
                ],
                cwd=TWIN,
            )

            if args.images_per_batch == 1:
                single_image = images / "cap_seal.png"
                if single_image.is_file():
                    single_image.rename(images / "cap_seal_0001.png")
            rendered_images = sorted(images.glob("cap_seal_*.png"))
            if len(rendered_images) != args.images_per_batch:
                raise RuntimeError(
                    f"expected {args.images_per_batch} renders in {images}, "
                    f"found {len(rendered_images)}"
                )
            for image_path in rendered_images:
                upscale_bottom_right(image_path)
                review_dir = html / f"{image_path.stem}_review"
                run(
                    [
                        args.labeler_python,
                        LABELER,
                        "run",
                        "--images", images,
                        "--files", image_path.name,
                        "--output", review_dir,
                        "--calibration", calibration,
                        "--profile", "capseal_bottom_right",
                        "--offset", 0, 0,
                    ],
                    cwd=REPO,
                )
                shutil.copyfile(review_dir / "index.html", html / f"{image_path.stem}.html")

            print(
                f"Batch {batch_number:03d}: {len(rendered_images)} images and "
                f"{len(rendered_images)} named review pages in {batch}",
                flush=True,
            )

    print(f"Completed {args.batches} batches at {output}", flush=True)


if __name__ == "__main__":
    main()
