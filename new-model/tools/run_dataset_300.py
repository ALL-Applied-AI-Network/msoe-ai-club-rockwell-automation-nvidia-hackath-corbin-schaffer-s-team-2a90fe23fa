"""Durable render/build/validate pipeline for the approved 300-scene dataset."""
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

repo=Path(__file__).resolve().parents[2]
run=repo/'new-model/dataset-300-run'
run.mkdir(exist_ok=True)
status=run/'status.json'
def state(stage, **extra):
    temp=status.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**extra),indent=2)+'\n')
    temp.replace(status)
def execute(args, log):
    with (run/log).open('w') as f:
        subprocess.run([str(a) for a in args],cwd=repo,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
    state('rendering',target_scenes=300,seed=20261030)
    execute([sys.executable,repo/'csi_digital_twin_v1/tools/isaac_headless.py',
             repo/'csi_digital_twin_v1/tools/render_example.py', '--camera','CapSealCam',
             '--stage','stations/tray_inspection/tray_inspection.usdc','--randomize-trays',
             '--variations','300','--min-vials','30','--max-vials','60',
             '--occupancy-mode','mixed','--seed','20261030','--export-placements',
             '--out',repo/'new-model/dataset-300-renders/capseal.png'],'render.log')
    renders=repo/'new-model/dataset-300-renders'
    assert len(list(renders.glob('*.placements.json')))==300
    state('building',scenes=300)
    execute([sys.executable,repo/'new-model/tools/build_dataset.py','--renders',renders,
             '--golden',repo/'new-model/GoldenImages/G24.jpg',
             '--output',repo/'new-model/dataset-300'],'build.log')
    dataset=repo/'new-model/dataset-300'
    state('validating',scenes=300)
    for split,mode in [('train','train'),('validation','evaluate'),('test','inference')]:
        execute([sys.executable,repo/'tao-skill-bank/skills/models/tao-train-visual-changenet/scripts/validate_vcn_dataset.py',
                 '--csv',dataset/'csv'/f'{split}.csv','--images-dir',dataset/'images',
                 '--mode',mode,'--batch-size','8'],f'validate-{split}.log')
    state('generated_pending_production_QA',scenes=300,dataset=str(dataset))
except BaseException:
    state('error',details=traceback.format_exc())
    raise
