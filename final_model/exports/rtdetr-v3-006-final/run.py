#!/usr/bin/env python3
"""Run the bundled model without the training repository."""
import argparse,hashlib,json,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--video',type=Path,default=ROOT/'examples/test_fill_000.mp4')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--threshold',type=float,default=.25)
p.add_argument('--confirm',type=int,default=3)
p.add_argument('--calibration',type=Path,default=ROOT/'calibration.json')
a=p.parse_args()
if a.output.exists():p.error('Output exists; choose a new directory')
if not a.video.is_file():p.error('Video does not exist')
if not 0<=a.threshold<=1:p.error('Threshold must be between 0 and 1')
if a.confirm<1:p.error('Confirmation must be positive')
if platform.machine() not in ('aarch64','arm64'):p.error('Bundled CUDA operator targets ARM64 GB10; rebuild/export for other systems')
if sys.version_info[:2]!=(3,12):p.error('Bundled operator requires Python 3.12')
manifest=json.loads((ROOT/'manifest.json').read_text())
for relative,expected in manifest['sha256'].items():
 path=ROOT/relative
 if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:p.error('Bundle checksum mismatch: '+relative)
import torch
if torch.__version__.split('+')[0]!='2.11.0' or torch.version.cuda!='13.0':p.error('Requires PyTorch 2.11.0 with CUDA 13.0')
if not torch.cuda.is_available():p.error('CUDA GPU is unavailable')
if torch.cuda.get_device_capability()!=(12,1):p.error('Bundled CUDA operator was built for GB10 sm_121; rebuild/export for another GPU')
command=[sys.executable,'-s',str(ROOT/'runtime/inspect_video.py'),'--video',str(a.video.resolve()),'--output',str(a.output.resolve()),'--calibration',str(a.calibration.resolve()),'--model',str(ROOT/'model/rtdetr-v3-006-final.ts'),'--threshold',str(a.threshold),'--confirm',str(a.confirm),'--device','cuda']
raise SystemExit(subprocess.call(command))
