"""Render one image from a camera of the digital twin with Isaac Sim (headless).

    ./python.sh tools/render_example.py --camera CapSealCam --out cap_seal.png
    ./python.sh tools/render_example.py --camera FillCam --frame 301 --out fill.png
    ./python.sh tools/render_example.py --camera CapSealCam --stage stations/tray_inspection/tray_inspection.usdc --out cap_seal.png
    ./python.sh tools/render_example.py --camera FillCam --stage stations/fill_station/fill_station.usda --out fill.png

Run with Isaac Sim's python (python.sh in the Isaac Sim install). By default it opens the full line
(csi_line.usda); --stage opens a single station file instead. The camera settings come from cameras/<name>.json.
CapSealCam is rendered with overscan and then given the real lens distortion (tools/apply_lens.py).
--randomize-trays varies the visibility and slot placement of existing CapSeal vials.
"""
import argparse, json, math, os, random, re, subprocess, sys, traceback
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--camera", required=True, choices=["CapSealCam", "FillCam"],
                help="camera name; settings come from cameras/<name>.json")
ap.add_argument("--stage", default=os.path.join(HERE, "csi_line.usda"), help="csi_line.usda or one station file")
ap.add_argument("--frame", type=int, default=1, help="time code (fill station animation: 1-686 at 30 fps)")
ap.add_argument("--out", default="render.png")
ap.add_argument("--subframes", type=int, default=32, help="path-tracing subframes (quality)")
ap.add_argument("--kit-arg", action="append", default=[], help="extra Isaac Sim / Kit argument (repeatable)")
ap.add_argument("--randomize-trays", action="store_true",
                help="randomize existing vial occupancy and placement across both CapSeal trays")
ap.add_argument("--variations", type=int, default=1,
                help="number of images to render; requires --randomize-trays when greater than 1")
ap.add_argument("--min-vials", type=int, default=1, help="minimum visible vials per randomized image")
ap.add_argument("--max-vials", type=int, default=None,
                help="maximum visible vials (defaults to all 60 tray slots)")
ap.add_argument("--broken-lean-degrees", type=float, default=0.0,
                help="maximum tilt from upright for broken S-series vials")
ap.add_argument("--seed", type=int, default=42, help="randomization seed for reproducible variations")
a = ap.parse_args()

cam = json.load(open(os.path.join(HERE, "cameras", a.camera + ".json")))
stage_path = Path(a.stage).expanduser()
if not stage_path.is_absolute():
    stage_path = Path(HERE) / stage_path
stage_path = str(stage_path.resolve())
if not os.path.isfile(stage_path):
    ap.error(f"stage does not exist: {stage_path}")
if a.frame < 1 or (a.camera == "FillCam" and a.frame > 686):
    ap.error("use a positive time code; the fill animation covers 1-686")
if a.subframes < 1:
    ap.error("--subframes must be positive")
if a.variations < 1:
    ap.error("--variations must be positive")
if a.variations > 1 and not a.randomize_trays:
    ap.error("--variations greater than 1 requires --randomize-trays")
if a.min_vials < 0 or (a.max_vials is not None and a.max_vials < 0):
    ap.error("--min-vials and --max-vials cannot be negative")
if not 0 <= a.broken_lean_degrees <= 3:
    ap.error("--broken-lean-degrees must be between 0 and 3 to avoid vials leaning into adjacent slots")
Path(a.out).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
a.out = str(Path(a.out).expanduser().resolve())
rw, rh = cam["render_px"]

from isaacsim import SimulationApp
kit = SimulationApp({"headless": True, "renderer": "PathTracing", "width": rw, "height": rh, "extra_args": a.kit_arg})
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import numpy as np, cv2
    import omni.usd, omni.timeline
    import omni.replicator.core as rep
    from pxr import Gf, Usd, UsdGeom

    ctx = omni.usd.get_context()
    if not ctx.open_stage(stage_path):
        raise RuntimeError(f"could not open stage: {stage_path}")
    while ctx.get_stage_loading_status()[2] > 0:
        kit.update()
    for _ in range(10):
        kit.update()
    stage = ctx.get_stage()
    prim = next((p for p in [cam["prim_in_csi_line"], cam["prim_in_station_file"]]
                 if stage.GetPrimAtPath(p).IsValid()), None)
    if prim is None:
        raise RuntimeError(f"camera {a.camera} not found in {stage_path}")

    stage.SetEditTarget(stage.GetSessionLayer())
    randomizer = random.Random(a.seed)
    randomize_trays = None
    if a.randomize_trays:
        if a.camera != "CapSealCam":
            ap.error("--randomize-trays is supported only with --camera CapSealCam")

        station_camera_path = cam["prim_in_station_file"]
        station_name = station_camera_path.strip("/").split("/", 1)[0]
        stage_camera_path = prim
        if not stage_camera_path.endswith(station_camera_path):
            raise RuntimeError(f"camera path {stage_camera_path} does not match {station_camera_path}")
        station_root_path = stage_camera_path[:-len(station_camera_path)] + "/" + station_name
        station_root = stage.GetPrimAtPath(station_root_path)
        if not station_root.IsValid():
            raise RuntimeError(f"tray inspection station not found at {station_root_path}")

        xform_cache = UsdGeom.XformCache()
        records = []
        good_tubes = []
        for child in station_root.GetChildren():
            name = child.GetName()
            if not re.fullmatch(r"[GS]\d+_tube", name):
                continue
            tube_position = xform_cache.GetLocalToWorldTransform(child).ExtractTranslation()
            tube_rotation = next(
                (op for op in UsdGeom.Xformable(child).GetOrderedXformOps()
                 if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ),
                None,
            )
            if tube_rotation is None:
                raise RuntimeError(f"no rotateXYZ operation on {child.GetPath()}")
            authored_rotation = tube_rotation.Get()
            damaged = name.startswith("S") or max(abs(float(authored_rotation[0])),
                                                   abs(float(authored_rotation[1]))) > 15
            cap = station_root.GetChild(name[:-5] + "_cap")
            cap_offset_local = None
            cap_rotation = None
            if cap.IsValid():
                cap_position = xform_cache.GetLocalToWorldTransform(cap).ExtractTranslation()
                cap_offset_local = xform_cache.GetLocalToWorldTransform(child).GetInverse().Transform(cap_position)
                cap_rotation = next(
                    (op for op in UsdGeom.Xformable(cap).GetOrderedXformOps()
                     if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ),
                    None,
                )
            records.append({
                "tube": child,
                "tube_position": tube_position,
                "tube_rotation": tube_rotation,
                "yaw": float(authored_rotation[2]),
                "cap": cap if cap.IsValid() else None,
                "cap_offset_local": cap_offset_local,
                "cap_rotation": cap_rotation,
                "broken": damaged,
            })
            if re.fullmatch(r"G\d+_tube", name):
                good_tubes.append(tube_position)

        def cluster_axis(values):
            clusters = []
            for value in sorted(float(value) for value in values):
                if not clusters or value - clusters[-1][-1] > 0.01:
                    clusters.append([value])
                else:
                    clusters[-1].append(value)
            return [sum(cluster) / len(cluster) for cluster in clusters]

        if not records or not good_tubes:
            raise RuntimeError("no existing CapSeal vial prims were found to randomize")
        slot_x = cluster_axis(position[0] for position in good_tubes)
        slot_y = cluster_axis(position[1] for position in good_tubes)
        if len(slot_x) * len(slot_y) != len(good_tubes):
            raise RuntimeError(
                f"could not infer a complete tray slot grid from {len(good_tubes)} good-tray vials"
            )

        good_tray = station_root.GetChild("tray_good_top")
        scrap_tray = station_root.GetChild("tray_scrap_top")
        if not good_tray.IsValid() or not scrap_tray.IsValid():
            raise RuntimeError("both good and scrap tray prims are required")
        good_center = xform_cache.GetLocalToWorldTransform(good_tray).ExtractTranslation()
        scrap_center = xform_cache.GetLocalToWorldTransform(scrap_tray).ExtractTranslation()
        slots = [(x, y) for x in slot_x for y in slot_y]
        slots.extend(
            (x + scrap_center[0] - good_center[0], y + scrap_center[1] - good_center[1])
            for x in slot_x for y in slot_y
        )
        if len(slots) != 2 * len(good_tubes):
            raise RuntimeError("the two trays' slot grid could not be determined")

        max_vials = len(slots) if a.max_vials is None else a.max_vials
        if max_vials > len(slots):
            ap.error(f"--max-vials cannot exceed the {len(slots)} available tray slots")
        if a.min_vials > max_vials:
            ap.error("--min-vials cannot exceed --max-vials")
        if max_vials > len(records):
            import omni.kit.commands

            source_records = list(records)
            for index in range(len(records), max_vials):
                source = randomizer.choice(source_records)
                source_name = source["tube"].GetName()
                kind = "S" if source["broken"] else "G"
                clone_base = f"Random{kind}_{index + 1:03d}"
                tube_path = f"{station_root_path}/{clone_base}_tube"
                success, _ = omni.kit.commands.execute(
                    "CopyPrimCommand",
                    path_from=str(source["tube"].GetPath()),
                    path_to=tube_path,
                )
                tube = stage.GetPrimAtPath(tube_path)
                if not tube.IsValid():
                    raise RuntimeError(f"failed to duplicate vial prim {source_name} to {tube_path}")
                cap = None
                cap_offset_local = None
                cap_rotation = None
                if source["cap"] is not None:
                    cap_path = f"{station_root_path}/{clone_base}_cap"
                    omni.kit.commands.execute(
                        "CopyPrimCommand",
                        path_from=str(source["cap"].GetPath()),
                        path_to=cap_path,
                    )
                    cap = stage.GetPrimAtPath(cap_path)
                    if not cap.IsValid():
                        raise RuntimeError(f"failed to duplicate cap prim to {cap_path}")
                    cap_offset_local = source["cap_offset_local"]
                    cap_rotation = next(
                        (op for op in UsdGeom.Xformable(cap).GetOrderedXformOps()
                         if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ),
                        None,
                    )
                tube_rotation = next(
                    (op for op in UsdGeom.Xformable(tube).GetOrderedXformOps()
                     if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ),
                    None,
                )
                if tube_rotation is None:
                    raise RuntimeError(f"no rotateXYZ operation on duplicated vial {tube_path}")
                records.append({
                    "tube": tube,
                    "tube_position": source["tube_position"],
                    "tube_rotation": tube_rotation,
                    "yaw": source["yaw"],
                    "cap": cap,
                    "cap_offset_local": cap_offset_local,
                    "cap_rotation": cap_rotation,
                    "broken": source["broken"],
                })
                UsdGeom.Imageable(tube).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                if cap is not None:
                    UsdGeom.Imageable(cap).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                xform_cache = UsdGeom.XformCache()
            print(f"Added {max_vials - len(source_records)} copied vial assemblies from existing USD objects",
                  flush=True)
        randomize_trays = (records, slots, max_vials)
        print(f"Tray randomization: {len(records)} vial assemblies, {len(slots)} total slots, "
              f"{a.min_vials}-{max_vials} vials per image", flush=True)

    tl = omni.timeline.get_timeline_interface()
    tl.set_current_time(a.frame / stage.GetTimeCodesPerSecond())
    kit.update()

    rp = rep.create.render_product(prim, (rw, rh))
    ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
    rep.orchestrator.set_capture_on_play(False)
    kit.update()

    # RTX's first sync of a scaled camera (CapSealCam carries the 0.0254 inch scale) misreads its
    # clippingRange and renders black. Changing and restoring the value in the session layer forces a
    # correct re-sync; the authored camera in the USD file is untouched.
    clip = UsdGeom.Camera(stage.GetPrimAtPath(prim)).GetClippingRangeAttr()
    near_far = clip.Get()
    clip.Set(Gf.Vec2f(near_far[0] * 0.5, near_far[1]))
    kit.update()
    clip.Set(near_far)
    kit.update()

    for variation in range(a.variations):
        output_path = a.out
        if a.variations > 1:
            output = Path(a.out)
            output_path = str(output.with_name(f"{output.stem}_{variation + 1:04d}{output.suffix}"))

        if randomize_trays is not None:
            records, slots, max_vials = randomize_trays
            count = randomizer.randint(a.min_vials, max_vials)
            chosen_records = randomizer.sample(records, count)
            chosen_slots = randomizer.sample(slots, count)

            for record in records:
                UsdGeom.Imageable(record["tube"]).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                if record["cap"] is not None:
                    UsdGeom.Imageable(record["cap"]).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

            def set_world_position(prim_to_move, position):
                parent = prim_to_move.GetParent()
                parent_to_world = xform_cache.GetLocalToWorldTransform(parent)
                local_position = parent_to_world.GetInverse().Transform(Gf.Vec3d(*position))
                translate_op = next(
                    (op for op in UsdGeom.Xformable(prim_to_move).GetOrderedXformOps()
                     if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
                    None,
                )
                if translate_op is None:
                    raise RuntimeError(f"no translate operation on {prim_to_move.GetPath()}")
                translate_op.Set(local_position)

            for record, (x, y) in zip(chosen_records, chosen_slots):
                tube_position = record["tube_position"]
                set_world_position(record["tube"], (x, y, tube_position[2]))
                yaw = record["yaw"]
                lean_x = 0.0
                lean_y = 0.0
                if record["broken"]:
                    lean = randomizer.uniform(0, a.broken_lean_degrees)
                    direction = math.radians(randomizer.uniform(0, 360))
                    lean_x = lean * math.cos(direction)
                    lean_y = lean * math.sin(direction)
                    yaw = randomizer.uniform(-180, 180)
                rotation = Gf.Vec3f(lean_x, lean_y, yaw)
                record["tube_rotation"].Set(rotation)
                if record["cap_rotation"] is not None:
                    cap_rotation = (
                        Gf.Vec3f(*(randomizer.uniform(0, 360) for _ in range(3)))
                        if record["broken"] else rotation
                    )
                    record["cap_rotation"].Set(cap_rotation)
                UsdGeom.Imageable(record["tube"]).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
                if record["cap"] is not None:
                    xform_cache.Clear()
                    tube_world = xform_cache.GetLocalToWorldTransform(record["tube"])
                    cap_position = tube_world.Transform(record["cap_offset_local"])
                    set_world_position(record["cap"], cap_position)
                    bounds_cache = UsdGeom.BBoxCache(
                        Usd.TimeCode.Default(),
                        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                    )
                    cap_bounds = bounds_cache.ComputeWorldBound(record["cap"]).ComputeAlignedRange()
                    tube_bounds = bounds_cache.ComputeWorldBound(record["tube"]).ComputeAlignedRange()
                    cap_min_z = cap_bounds.GetMin()[2]
                    tube_max_z = tube_bounds.GetMax()[2]
                    lift = max(0.0, tube_max_z - cap_min_z)
                    if lift:
                        set_world_position(record["cap"], (cap_position[0], cap_position[1],
                                                            cap_position[2] + lift))
                    UsdGeom.Imageable(record["cap"]).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            print(f"Variation {variation + 1}: {count} vials", flush=True)

        kit.update()
        for _ in range(2):
            rep.orchestrator.step(rt_subframes=a.subframes, delta_time=0.0)
        data = np.asarray(ann.get_data())
        if data.ndim != 3 or data.shape[:2] != (rh, rw) or data.shape[2] < 3:
            raise RuntimeError(f"RGB capture returned shape {data.shape}; check GPU/RTX startup errors")
        img = data[..., :3].copy()

        # Save and post-process before close(): Isaac's default fast shutdown exits
        # the process and will never execute statements after kit.close().
        if cam.get("overscan", 1.0) != 1.0:
            tmp = output_path + ".pinhole.png"
            if not cv2.imwrite(tmp, img[..., ::-1]):
                raise OSError(f"could not write {tmp}")
            try:
                subprocess.run([sys.executable, os.path.join(HERE, "tools", "apply_lens.py"), tmp,
                                os.path.join(HERE, "cameras", a.camera + ".json"), output_path], check=True)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        elif not cv2.imwrite(output_path, img[..., ::-1]):
            raise OSError(f"could not write {output_path}")
        print("wrote", output_path, flush=True)
except BaseException:
    traceback.print_exc()
    # Preserve failure status even when Kit's fast shutdown terminates Python.
    kit.close(exit_code=1)
    raise
kit.close()
