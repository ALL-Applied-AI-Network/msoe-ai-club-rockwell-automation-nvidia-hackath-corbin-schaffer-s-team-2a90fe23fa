#!/usr/bin/env python3
"""Run Isaac Python with the NVIDIA EGL Vulkan ICD verified on this GB10."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
isaac = root / 'IsaacSim/_build/linux-aarch64/release/python.sh'
library = Path('/lib/aarch64-linux-gnu/libEGL_nvidia.so.0')
if not isaac.is_file() or not library.is_file():
    sys.exit('Expected local Isaac Sim and NVIDIA EGL library were not found')
with tempfile.TemporaryDirectory(prefix='isaac-egl-') as temp:
    icd = Path(temp) / 'nvidia.json'
    icd.write_text(json.dumps({'file_format_version': '1.0.1', 'ICD': {
        'library_path': str(library), 'api_version': '1.4.312'}}))
    env = os.environ.copy()
    for key in ('CONDA_PREFIX', 'LD_LIBRARY_PATH', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(VK_ICD_FILENAMES=str(icd), VK_DRIVER_FILES=str(icd))
    result = subprocess.run([str(isaac), *sys.argv[1:]], env=env)
    sys.exit(result.returncode)
