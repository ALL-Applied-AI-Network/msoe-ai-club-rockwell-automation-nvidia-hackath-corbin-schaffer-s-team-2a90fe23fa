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
from cap_label_policy import label_cap, POLICY

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
ap.add_argument("--export-placements", action="store_true", help="save per-frame slot identity sidecars")
ap.add_argument("--occupancy-mode", choices=["random", "mixed"], default="random",
                help="mixed cycles full trays, valid fill prefixes and random gaps")
ap.add_argument("--placements-only", action="store_true", help="replay seed to export updated labels without rendering")
ap.add_argument("--verify-placements-dir", type=Path, help="require replay identity and bounds to match existing sidecars")
ap.add_argument("--segment-pairs", type=Path, help="export controlled full-frame PNG pairs under this dataset root")
ap.add_argument("--segment-defect", choices=["tilt", "missing", "random", "none"], default="tilt")
ap.add_argument("--segment-tilt-degrees", type=float, default=15.0)
ap.add_argument("--segment-unchanged-every", type=int, default=5,
                help="every Nth pair is normal/normal; 0 disables unchanged pairs")
ap.add_argument("--export-detections", action="store_true", help="export full-frame defect boxes with slot IDs; implies placement export")
ap.add_argument("--detection-scope", choices=["G", "both"], default="G", help="prioritize good tray G01-G30 by default")
a = ap.parse_args()
if a.export_detections:
    if a.segment_pairs or a.placements_only or a.camera != "CapSealCam":
        ap.error("detection export requires rendered randomized CapSeal frames, separate from segment pairs")
    a.export_placements = True
    a.randomize_trays = True
if a.segment_pairs:
    if a.camera != "CapSealCam" or a.placements_only or a.export_placements:
        ap.error("segment pairs require CapSealCam and cannot use placements-only/export-placements")
    if a.min_vials < 1 or not 0 < a.segment_tilt_degrees < 90 or a.segment_unchanged_every < 0:
        ap.error("segment pairs require min-vials >=1, tilt in (0,90), and unchanged-every >=0")
    a.randomize_trays = True
    if a.broken_lean_degrees:
        ap.error("segment pairs require upright vials; omit broken-lean-degrees")
    if a.segment_pairs.exists():
        ap.error("segment dataset root already exists; choose a new directory")
    for directory in ("A", "B", "list", "scenes"):
        (a.segment_pairs / directory).mkdir(parents=True, exist_ok=True)

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
                "source_id": name,
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

        if a.segment_pairs:
            # Use only authored normal assemblies: damaged-source cap offsets are not normal references.
            records = [r for r in records if not r["broken"] and r["cap"] is not None
                       and r["cap_rotation"] is not None]
            if not records:
                raise RuntimeError("no normal capped assemblies available for segment pairs")
            retained = {str(r["tube"].GetPath()) for r in records}
            for child in station_root.GetChildren():
                if re.fullmatch(r"[GS]\d+_(tube|cap)", child.GetName()):
                    tube_path = str(child.GetPath()).removesuffix("_cap") + "_tube" if child.GetName().endswith("_cap") else str(child.GetPath())
                    if tube_path not in retained:
                        UsdGeom.Imageable(child).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

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
                    "source_id": source["source_id"],
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
            occupancy_kind = "random_gaps"
            if a.occupancy_mode == "mixed" and variation % 3 == 0:
                count = max_vials
                occupancy_kind = "full" if count == 60 else "random_gaps"
            chosen_records = randomizer.sample(records, count)
            chosen_slots = randomizer.sample(slots, count)
            if a.occupancy_mode == "mixed" and variation % 3 == 1:
                occupancy_kind = "valid_prefix"
                def projected_order(xy):
                    world = np.asarray([*xy, float(records[0]["tube_position"][2])])
                    xyz = np.asarray(cam["R_world_to_cv"]) @ (world-np.asarray(cam["position_m"]))
                    return xyz[:2]/xyz[2]
                ordered_trays = []
                for offset in (0,30):
                    tray = slots[offset:offset+30]
                    columns = sorted({xy[0] for xy in tray},
                                     key=lambda x: projected_order((x,tray[0][1]))[0], reverse=True)
                    ordered_trays.append([xy for x in columns for xy in sorted(
                        [p for p in tray if p[0]==x],key=lambda p: projected_order(p)[1])])
                first_count = randomizer.randint(max(0,count-30),min(30,count))
                chosen_slots = ordered_trays[0][:first_count]+ordered_trays[1][:count-first_count]
            placed_records = dict(zip(chosen_slots, chosen_records))

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

        def capture(output_path):
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
        if a.segment_pairs:
            name = f"pair_{variation + 1:06d}.png"
            target = chosen_records[variation % len(chosen_records)]
            unchanged = (a.segment_defect == "none" or
                         (a.segment_unchanged_every > 0 and (variation + 1) % a.segment_unchanged_every == 0))
            defect = a.segment_defect
            if unchanged:
                defect = "none"
            elif a.segment_defect == "random":
                defect = random.choice(["tilt", "missing"])
            root = a.segment_pairs
            scene_file = root / "scenes" / f"{Path(name).stem}.usda"
            # Flatten A so the complete normal scene (camera, lights, placements) is reproducible.
            if not stage.Export(str(scene_file)):
                raise OSError(f"could not export normal scene to {scene_file}")
            normal_rotation = target["cap_rotation"].Get()
            normal_visibility = UsdGeom.Imageable(target["cap"]).GetVisibilityAttr().Get()
            transforms = []
            xform_cache.Clear()
            for record in chosen_records:
                transforms.append({"tube": str(record["tube"].GetPath()),
                                   "cap": str(record["cap"].GetPath()),
                                   "tube_world": [list(row) for row in xform_cache.GetLocalToWorldTransform(record["tube"])],
                                   "cap_world": [list(row) for row in xform_cache.GetLocalToWorldTransform(record["cap"])]})
            try:
                capture(str(root / "A" / name))
                if defect == "tilt":
                    target["cap_rotation"].Set(Gf.Vec3f(a.segment_tilt_degrees + random.randint(0, 25), 0, normal_rotation[2]))
                elif defect == "missing":
                    UsdGeom.Imageable(target["cap"]).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                capture(str(root / "B" / name))
            finally:
                target["cap_rotation"].Set(normal_rotation)
                UsdGeom.Imageable(target["cap"]).GetVisibilityAttr().Set(normal_visibility)
                xform_cache.Clear()
            metadata = {"image": name, "seed": a.seed, "variation": variation + 1,
                        "camera": cam, "stage": stage_path, "normal_scene": str(scene_file),
                        "frame": a.frame, "subframes": a.subframes, "assemblies": transforms,
                        "target_cap": str(target["cap"].GetPath()), "defect": defect,
                        "tilt_degrees": a.segment_tilt_degrees if defect == "tilt" else 0,
                        "crop": "full frame; identical lens processing for A and B",
                        "ground_truth_masks": False}
            (root / "scenes" / f"{Path(name).stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
            with (root / "list" / "test.txt").open("a") as listing:
                listing.write(name + "\n")
            print(f"wrote segment pair {name} ({defect})", flush=True)
        elif not a.placements_only:
            capture(output_path)
        if a.export_placements:
            if randomize_trays is None or a.camera != "CapSealCam":
                raise RuntimeError("placement export requires randomized CapSeal trays")
            # Projection matches the lens-corrected 640x480 image, before review resizing.
            def project_point(world):
                xyz = np.asarray(cam["R_world_to_cv"]) @ (np.asarray(world) - np.asarray(cam["position_m"]))
                if xyz[2] <= 0:
                    raise RuntimeError("slot behind camera")
                x, y = xyz[:2] / xyz[2]
                distortion = 1 + cam["lens_k1"] * (x*x + y*y)
                return [float(x * distortion * cam["focal_px_final"] + cam["principal_point_final_px"][0]),
                        float(y * distortion * cam["focal_px_final"] + cam["principal_point_final_px"][1])]
            reference_z = float(records[0]["tube_position"][2])
            entries = []
            for tray_index, prefix in enumerate(("G", "S")):
                tray_slots = slots[tray_index*30:(tray_index+1)*30]
                # Camera X runs right-to-left relative to world X. Sort columns in image space.
                columns = sorted({xy[0] for xy in tray_slots},
                                 key=lambda x: project_point((x, tray_slots[0][1], reference_z))[0], reverse=True)
                ordered = []
                for x in columns:
                    ordered.extend(sorted([xy for xy in tray_slots if xy[0] == x],
                                          key=lambda xy: project_point((*xy, reference_z))[1]))
                occupied = [i+1 for i, xy in enumerate(ordered) if xy in placed_records]
                boundary = max(occupied, default=0)
                for position, xy in enumerate(ordered, 1):
                    record = placed_records.get(xy)
                    center = project_point((*xy, reference_z))
                    inspection_center = None
                    inspection_bounds = None
                    cap_policy = {"label": "no_vial", "cap_tilt_degrees": None, "label_reason": "empty_slot"}
                    if record:
                        xform_cache.Clear()
                        vial_axis = xform_cache.GetLocalToWorldTransform(record["tube"]).TransformDir(Gf.Vec3d(0,0,1))
                        cap_axis = (xform_cache.GetLocalToWorldTransform(record["cap"]).TransformDir(Gf.Vec3d(0,0,1))
                                    if record["cap"] is not None else (0,0,1))
                        cap_policy = label_cap(record["cap"] is not None, vial_axis, cap_axis)
                        # Inspect the vial mouth/cap, not the assembly's base transform.
                        # Use world bounds after randomization; cap offsets differ by pose.
                        geometry_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
                        tube_range = geometry_cache.ComputeWorldBound(record["tube"]).ComputeAlignedRange()
                        lo, hi = tube_range.GetMin(), tube_range.GetMax()
                        mouth = ((lo[0]+hi[0])/2, (lo[1]+hi[1])/2, hi[2])
                        points = [project_point((x, y, hi[2]))
                                  for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]
                        inspection_center = project_point(mouth)
                        if record["cap"] is not None:
                            cap_range = geometry_cache.ComputeWorldBound(record["cap"]).ComputeAlignedRange()
                            clo, chi = cap_range.GetMin(), cap_range.GetMax()
                            points.extend(project_point((x,y,z)) for x in (clo[0],chi[0])
                                          for y in (clo[1],chi[1]) for z in (clo[2],chi[2]))
                        inspection_bounds = [min(p[0] for p in points), min(p[1] for p in points),
                                             max(p[0] for p in points), max(p[1] for p in points)]
                    entries.append({"slot_id": f"{prefix}{position:02}", "tray": prefix,
                                    "position": position, "center_native": center,
                                    "inspection_center_native": inspection_center,
                                    "inspection_bounds_native": inspection_bounds,
                                    "world_position": [float(xy[0]), float(xy[1]), reference_z],
                                    "present": record is not None,
                                    "gap_error": record is None and position < boundary,
                                    **cap_policy, "cap_present": record is not None and record["cap"] is not None,
                                    "label_source": POLICY, "cap_tilt_threshold_degrees": 1.0,
                                    "source_id": record["source_id"] if record else None,
                                    "instance_id": str(record["tube"].GetPath()) if record else None})
            if a.verify_placements_dir:
                previous = json.loads((a.verify_placements_dir / Path(output_path).with_suffix(".placements.json").name).read_text())
                for old, new in zip(previous["slots"], entries):
                    for key in ("slot_id", "source_id", "instance_id", "present", "gap_error"):
                        if old[key] != new[key]:
                            raise RuntimeError(f"Replay mismatch: {key} for {new['slot_id']}")
                    for key in ("center_native", "inspection_bounds_native"):
                        if old[key] is None and new[key] is None:
                            continue
                        if old[key] is None or new[key] is None or any(abs(x-y)>1e-5 for x,y in zip(old[key],new[key])):
                            raise RuntimeError(f"Replay geometry mismatch for {new['slot_id']}")
            Path(output_path).with_suffix(".placements.json").write_text(json.dumps({
                "image": Path(output_path).name, "seed": a.seed, "variation": variation+1,
                "camera": a.camera, "stage": stage_path, "native_size": [640,480],
                "fill_order": "top_to_bottom_then_left_per_tray",
                "occupancy_kind": occupancy_kind, "slots": entries}, indent=2)+"\n")
        if a.export_detections:
            from full_frame_detection import frame_annotations
            meta = json.loads(Path(output_path).with_suffix(".placements.json").read_text())
            detections = frame_annotations(meta, a.detection_scope)
            Path(output_path).with_suffix(".detections.json").write_text(json.dumps(detections, indent=2)+"\n")
        if not a.segment_pairs:
            print("wrote", output_path, flush=True)
except BaseException:
    traceback.print_exc()
    # Preserve failure status even when Kit's fast shutdown terminates Python.
    kit.close(exit_code=1)
    raise
kit.close()
