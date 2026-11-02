#!/usr/bin/env python3
"""Run the Windows x64 CPU ONNX vial-inspection bundle."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from runtime.full_frame_detection import assign_to_slot
from runtime.temporal import VideoInspector


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model/rtdetr-v3-006-final.onnx"
CALIBRATION = ROOT / "calibration.json"


def verify_bundle() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Bundle file is missing: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Bundle checksum mismatch: {relative}")


def inspect_video(video: Path, output: Path, calibration_path: Path, threshold: float, confirm: int) -> None:
    if not video.is_file():
        raise FileNotFoundError(f"Video does not exist: {video}")
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    slots = calibration["slots"]
    tracker = VideoInspector(slots, confirm, confirm, threshold)
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"Expected CPU-only inference, got providers: {session.get_providers()}")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise ValueError("Video has no valid FPS")
    output.mkdir(parents=True)
    writer = None
    frame_count = 0
    complete_frames = 0
    try:
        with (output / "frame-results.jsonl").open("w", encoding="utf-8") as results:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                if [width, height] != calibration["image_size"]:
                    raise ValueError(
                        f"Video is {width}x{height}, calibration expects "
                        f"{calibration['image_size'][0]}x{calibration['image_size'][1]}"
                    )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255
                raw = session.run(["detections"], {"image": tensor})[0][0]
                detections = []
                for x0, y0, x1, y1, score, category in raw:
                    category_id = int(category)
                    if category_id not in (1, 2, 3):
                        continue
                    detections.append(
                        {
                            "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                            "score": float(score),
                            "category_id": category_id,
                        }
                    )
                assigned = []
                for detection in detections:
                    slot = assign_to_slot(detection["bbox"], slots, "G")
                    if slot:
                        assigned.append(dict(detection, slot_id=slot))
                status = tracker.update(assigned, frame_count, frame_count / fps)
                results.write(json.dumps(status) + "\n")
                complete_frames += status["order_status"] == "complete"
                cv2.putText(
                    frame,
                    f"G: {status['occupied_count']}/30 {status['order_status']}",
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0) if status["order_status"] == "complete" else (0, 180, 255),
                    2,
                )
                for detection in assigned:
                    if detection["category_id"] == 1 or detection["score"] < threshold:
                        continue
                    x, y, width_box, height_box = map(round, detection["bbox"])
                    cv2.rectangle(frame, (x, y), (x + width_box, y + height_box), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"{detection['slot_id']} {detection['score']:.2f}",
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        1,
                    )
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(output / "inspection.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Cannot create output video")
                writer.write(frame)
                frame_count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if frame_count == 0:
        raise ValueError("Video contained no decoded frames")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "frames": frame_count,
                "fps": fps,
                "complete_frames": complete_frames,
                "mode": "onnxruntime_cpu",
                "model": MODEL.name,
                "threshold": threshold,
                "confirmation_frames": confirm,
                "note": "Learned detector used; counts are outputs, not ground-truth accuracy scores.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--confirm", type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("Threshold must be between 0 and 1.")
    if args.confirm < 1:
        parser.error("Confirmation frames must be positive.")
    verify_bundle()
    inspect_video(args.video.resolve(), args.output.resolve(), args.calibration.resolve(), args.threshold, args.confirm)


if __name__ == "__main__":
    main()
