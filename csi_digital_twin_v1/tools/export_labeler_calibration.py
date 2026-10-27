"""Export vial-labeler slot calibration for the CapSeal render and bottom-right crop."""
import argparse
import json
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True)
ap.add_argument("--camera-json", required=True)
ap.add_argument("--output", required=True)
args = ap.parse_args()

app = SimulationApp({"headless": True})
try:
    import omni.usd
    from pxr import UsdGeom

    camera = json.loads(Path(args.camera_json).read_text(encoding="utf-8"))
    final_width, final_height = camera["final_image_px"]
    target_width, target_height = 1920, 1080
    scale = max(target_width / final_width, target_height / final_height)
    resized_width = round(final_width * scale)
    resized_height = round(final_height * scale)
    crop_left = resized_width - target_width
    crop_top = resized_height - target_height

    context = omni.usd.get_context()
    if not context.open_stage(args.stage):
        raise RuntimeError(f"could not open stage: {args.stage}")
    while context.get_stage_loading_status()[2] > 0:
        app.update()
    stage = context.get_stage()
    root = stage.GetPrimAtPath("/TrayInspection")
    good_tray = root.GetChild("tray_good_top")
    scrap_tray = root.GetChild("tray_scrap_top")
    if not root.IsValid() or not good_tray.IsValid() or not scrap_tray.IsValid():
        raise RuntimeError("stage must contain the CapSeal tray inspection prims")

    cache = UsdGeom.XformCache()
    def matrix_array(matrix):
        return np.array(
            [[float(matrix[row][column]) for column in range(4)] for row in range(4)],
            dtype=float,
        )

    good_matrix = matrix_array(cache.GetLocalToWorldTransform(good_tray))
    scrap_matrix = matrix_array(cache.GetLocalToWorldTransform(scrap_tray))
    good_to_scrap = np.linalg.inv(good_matrix) @ scrap_matrix

    positions = {}
    for child in root.GetChildren():
        name = child.GetName()
        if name.startswith("G") and name.endswith("_tube"):
            positions[name] = np.array(
                [*cache.GetLocalToWorldTransform(child).ExtractTranslation(), 1.0],
                dtype=float,
            )
    if len(positions) != 30:
        raise RuntimeError(f"expected 30 good-tray slot references, found {len(positions)}")

    rotation = np.asarray(camera["R_world_to_cv"], dtype=float)
    camera_position = np.asarray(camera["position_m"], dtype=float)
    focal = float(camera["focal_px_final"])
    distortion = float(camera["lens_k1"])
    principal_x, principal_y = camera["principal_point_final_px"]

    def project(world):
        camera_xyz = rotation @ (world[:3] - camera_position)
        if camera_xyz[2] <= 0:
            raise RuntimeError("a tray slot projects behind the camera")
        x = camera_xyz[0] / camera_xyz[2]
        y = camera_xyz[1] / camera_xyz[2]
        radius_squared = x * x + y * y
        x *= 1 + distortion * radius_squared
        y *= 1 + distortion * radius_squared
        return [
            x * focal * scale + (principal_x + 0.5) * scale - 0.5 - crop_left,
            y * focal * scale + (principal_y + 0.5) * scale - 0.5 - crop_top,
        ]

    def tray_grid(prefix):
        projected = {}
        for column in range(6):
            for row in range(5):
                position = column * 5 + row + 1
                world = positions[f"G{position:02}_tube"]
                if prefix == "S":
                    world = world @ good_to_scrap
                projected[position] = project(world)
        return [
            projected[1],
            projected[26],
            projected[5],
            projected[30],
        ]

    calibration = {
        "image_size": [target_width, target_height],
        "max_alignment_error": 0.3,
        "profiles": [
            {
                "id": "capseal_bottom_right",
                "reference_image": "",
                "reference_sha256": "",
                "alignment_anchors": [],
                "areas": {
                    "good": {
                        "grid": tray_grid("G"),
                        "crop_offsets": [-45, -54, 45, 54],
                    },
                    "scrap": {
                        "grid": tray_grid("S"),
                        "crop_offsets": [-45, -54, 45, 54],
                    },
                },
            }
        ],
    }
    Path(args.output).write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote CapSeal calibration for {target_width}x{target_height}: {args.output}", flush=True)
finally:
    app.close()
