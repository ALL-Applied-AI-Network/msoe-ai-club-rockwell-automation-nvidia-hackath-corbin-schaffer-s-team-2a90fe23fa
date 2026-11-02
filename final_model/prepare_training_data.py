"""Generate and validate an expanded video dataset. Never starts a training job."""
import json,os,subprocess,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
root=Path(__file__).resolve().parents[1];os.chdir(root)
run=root/'final_model/preparation-v3';run.mkdir(exist_ok=True)
status=run/'status.json';dataset=root/'final_model/dataset-v3'
def state(stage,**kw):
 p=status.with_suffix('.tmp');p.write_text(json.dumps(dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),training_started=False,**kw),indent=2)+'\n');p.replace(status)
def execute(command,log):
 with (run/log).open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 state('GENERATING',planned_sequences={'train':60,'validation':15,'test':15})
 execute([sys.executable,'-s','final_model/generate_dataset.py','--output',str(dataset),
          '--train-sequences','60','--validation-sequences','15','--test-sequences','15',
          '--seed-base','110000','--vary-sequences',
          '--train-defect-probability','0.5','--eval-defect-probability','0.1'],'generation.log')
 state('VALIDATING')
 execute([sys.executable,'-s','final_model/validate_dataset.py','--dataset',str(dataset)],'validation.log')
 validation=json.loads((dataset/'validation.json').read_text());coverage={}
 for split in ('train','validation','test'):
  data=json.loads((dataset/'annotations'/f'{split}-keyframes.json').read_text())
  counts=Counter(a['category_id'] for a in data['annotations'])
  coverage[split]={'keyframes':len(data['images']),'annotations_by_class':dict(counts)}
  assert all(counts[k]>0 for k in (1,2,3)),f'{split}: missing category'
 # Counts are a baseline readiness check, not proof of generalization.
 state('DATA_READY_FOR_DETECTOR_SMOKE',dataset=str(dataset),coverage=coverage,
       defect_rates=validation.get('defect_rates'),
       note='All sequence/video/oracle checks passed. No training launched. Native detector runtime/backbone/configuration still needs smoke testing; real-world quality unverified.')
except BaseException:
 state('ERROR',traceback=traceback.format_exc());raise
