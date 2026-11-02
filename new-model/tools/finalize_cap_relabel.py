"""Finish verified replay without overwriting the original dataset."""
import json, time, subprocess, sys, os
from pathlib import Path
root=Path(__file__).resolve().parents[2]
os.chdir(root)
labels=root/'new-model/dataset-300-relabel'
status=labels/'status.json'
def state(stage, **kwargs):
    status.write_text(json.dumps(dict(stage=stage,**kwargs),indent=2)+'\n')
try:
    state('replaying_and_verifying',target=300)
    for _ in range(900):
        files=list(labels.glob('*.placements.json'))
        if len(files)==300:break
        time.sleep(2)
    else:raise RuntimeError('Replay did not produce 300 verified sidecars within 30 minutes; inspect /tmp/vial-relabel.log')
    changed=0;counts={}
    for p in files:
        previous=json.loads((root/'new-model/dataset-300-renders'/p.name).read_text())
        current=json.loads(p.read_text())
        for old,new in zip(previous['slots'],current['slots']):
            changed+=old['label']!=new['label']
            counts[new['label']]=counts.get(new['label'],0)+1
        image=labels/current['image']
        if not image.exists():os.link(root/'new-model/dataset-300-renders'/current['image'],image)
    state('building',changed_labels=changed,slot_label_counts=counts)
    subprocess.run([sys.executable,'new-model/tools/build_dataset.py','--renders',str(labels),
                    '--output','new-model/dataset-300-tilt'],check=True)
    for split,mode in [('train','train'),('validation','evaluate'),('test','inference')]:
        subprocess.run([sys.executable,'tao-skill-bank/skills/models/tao-train-visual-changenet/scripts/validate_vcn_dataset.py',
                        '--csv',f'new-model/dataset-300-tilt/csv/{split}.csv','--images-dir','new-model/dataset-300-tilt/images',
                        '--mode',mode,'--batch-size','8'],check=True)
    state('relabeled_validated_pending_visibility_QA',changed_labels=changed,slot_label_counts=counts)
except BaseException as error:
    state('error',reason=str(error));raise
