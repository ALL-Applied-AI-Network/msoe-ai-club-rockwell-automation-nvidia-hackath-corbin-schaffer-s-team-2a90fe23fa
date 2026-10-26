# CSI vial line: digital twin

This is a digital twin of the CSI vial line, prepared for generating synthetic data in NVIDIA Isaac Sim / Omniverse. It contains two stations, each with a camera matched to a real camera on the line:

- **Tray inspection station:** robot, inspection trays and capped vials.
- **Fill (nozzle) station:** vial carts under the dispensing nozzle, with a filling animation.

Units are **meters** and **Z is up**. There is **no physics** anywhere: everything is static geometry, except the fill station's pre-recorded filling animation.

## Contents
```
csi_line.usda                        the full line (open this for everything)
stations/
  tray_inspection/tray_inspection.usdc   tray inspection station + cap-seal camera + its lights
  fill_station/fill_station.usda         fill station + fill camera + its lights (animated, frames 1-686)
  magnemover/magnemover.usdc             MagneMover track and carts
  facility/facility.usdc                 the rest of the plant, floor and dome light
cameras/                             camera settings (JSON), one file per camera
tools/
  render_example.py                  render one image from any camera (Isaac Sim)
  apply_lens.py                      adds the real cap-seal camera lens distortion to a render
```
Keep the folder structure as it is: the files reference each other by relative paths.

## The full line or a single station?
`csi_line.usda` **is** the digital twin. It does not hold any geometry itself: it references the four station files and places them together.

| You want to... | Open |
|---|---|
| See or render the whole line | `csi_line.usda` |
| Work only on the tray inspection data | `stations/tray_inspection/tray_inspection.usdc` |
| Work only on the fill station data | `stations/fill_station/fill_station.usda` |

- **Open a station file on its own** when you only need that station. It loads in seconds, while the full line loads the whole plant (about 0.7 GB). Each station file has its own camera(s) and lights, so you can render from it directly.
- **To change something inside a station** (a material, a vial, a light), **edit the station file.** The full line picks up the change automatically.
- **Edit `csi_line.usda` only for line-level changes:** where the stations sit relative to each other, or switching parts of the line on or off.

  If you edit a station's contents while `csi_line.usda` is open, Isaac Sim saves the change into `csi_line.usda` as an override. It applies only to the full line, not to the station file.
- The tray inspection, MagneMover and facility files use the same coordinates as the full line. The fill station file uses its own local coordinates, and `csi_line.usda` moves it into place on the track.

## Cameras
| Camera | Station | Final image | Prim path (in the station file) |
|---|---|---|---|
| `CapSealCam` | tray inspection | 640 × 480 | `/TrayInspection/CapSealCam/CapSealCam` |
| `FillCam` | fill station | 640 × 480 | `/FillStation/FillStation_LocalRoot/FillCam/FillCam` |

- **Paths in the full line:** add the prefix `/CSI_Line`. For example, `/CSI_Line/TrayInspection/CapSealCam/CapSealCam`.
- **Camera JSON files:** `cameras/<name>.json` has each camera's position, orientation, focal length, resolution and lens values, plus its prim path in both the station file and the full line.
- **Cap-seal camera:** `CapSealCam` matches the real 640 × 480 cap-seal camera, solved from the cap_seal video clip.
- **Lens distortion (CapSealCam):** the real camera has a slight lens distortion, so:
  1. Render at `render_px` (768 × 576, a 1.2× wider pinhole view).
  2. Run `tools/apply_lens.py`, which adds the distortion and crops to 640 × 480.
- **Fill camera:** a plain pinhole. Render 640 × 480 directly.

## Fill station animation
- The fill station plays a filling sequence over time codes **1–686** at **30 fps** (22.9 s). Blue liquid fills, then clear liquid is added, while the next cart moves in.
- Pick the moment by time code. For example, 301 is mid-fill and 686 is the finished vial.
- Everything else in the line is static.

## Quick start
Run with Isaac Sim's Python (`python.sh` in the Isaac Sim install folder):
```bash
# full line, fill camera, finished vial
<isaac-sim>/python.sh tools/render_example.py --camera FillCam --frame 686 --out fill.png
# full line, cap-seal camera (lens distortion applied automatically)
<isaac-sim>/python.sh tools/render_example.py --camera CapSealCam --out cap_seal.png
# a single station file
<isaac-sim>/python.sh tools/render_example.py --camera FillCam --frame 301 \
    --stage stations/fill_station/fill_station.usda --out fill_station.png
# 20 CapSealCam images with randomized vial counts and placements, up to 60
<isaac-sim>/python.sh tools/render_example.py --camera CapSealCam --randomize-trays \
    --variations 20 --min-vials 1 --max-vials 60 --broken-lean-degrees 0 \
    --seed 42 --out cap_seal.png
```
For multiple variations, the output is numbered (`cap_seal_0001.png`, etc.). The randomizer
uses the 35 existing vial assemblies (including existing uncapped examples) and creates copies of
those assemblies as needed to support all 60 tray slots. It hides unused vials and places selected
vials into distinct slots across both trays. The camera stays fixed. CapSeal renders are
lens-corrected automatically.
Change `--seed` for a different reproducible sequence; set `--min-vials` and `--max-vials` to
control the randomized vial count. To fill both trays completely in every image, set both counts to
60, for example `--min-vials 60 --max-vials 60`.
Existing `S`-series and any other strongly tilted vial variants are kept, but their original steep
tilt is replaced by an upright pose by default so they do not intrude into adjacent vial slots.
All randomized vial variants are placed upright, preserving their yaw; caps follow the vial's
local transform, keeping them seated when a vial is repositioned or straightened. Use
`--broken-lean-degrees` to add a small randomized lean (allowed range: 0–3 degrees).
Caps on non-good (`S`-series or strongly tilted) vials additionally receive independent
randomized 0–360° rotations around X, Y, and Z; caps on good vials remain aligned.
Rotated caps are lifted as needed so their bounds do not extend below the top of their vial.

`apply_lens.py` can also be run on its own, with any Python that has `numpy` and `opencv-python`:
```bash
python tools/apply_lens.py cap_seal_render_768x576.png cameras/CapSealCam.json cap_seal_final.png
```

## Notes
- **Materials** are standard USD materials (UsdPreviewSurface). Glass and liquid are approximations, so you may want to adjust them for Isaac Sim's RTX renderer.
- **Labels on fill station parts:** they carry `userProperties:synthetic_class`, `synthetic_class_id` and `synthetic_instance_id` attributes, which can be used as labels.
- **Tested with:** Isaac Sim 6.0.
