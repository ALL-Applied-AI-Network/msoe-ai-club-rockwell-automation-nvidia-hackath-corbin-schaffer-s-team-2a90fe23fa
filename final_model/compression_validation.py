"""Separate fixed-cutoff clean/compressed validation measurements during training."""
import json,sys
from pathlib import Path
from collections import Counter,defaultdict
import cv2,torch
from pytorch_lightning import Callback
from detector_adapter import RTDETRVideoAdapter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'csi_digital_twin_v1/tools'))
from full_frame_detection import assign_to_slot

class CompressionValidation(Callback):
 def __init__(self,dataset,output,num_select,target_epochs):
  self.dataset=dataset;self.output=output;self.num_select=num_select;self.target_epochs=target_epochs
  self.slots=json.loads((dataset/'calibration.json').read_text())['slots']
 def on_validation_end(self,trainer,pl_module):
  if trainer.sanity_checking:return
  epoch=trainer.current_epoch+1
  if not trainer.fast_dev_run and epoch!=28 and epoch%10 and epoch!=self.target_epochs:return
  was_training=pl_module.model.training
  adapter=RTDETRVideoAdapter(pl_module.model.eval(),num_select=self.num_select)
  result={'epoch':epoch,'global_step':trainer.global_step,'threshold':.25,'variants':{}}
  try:
   for variant in ('clean','compressed'):
    coco=json.loads((self.dataset/f'annotations/validation-{variant}.json').read_text());anns=defaultdict(list)
    for a in coco['annotations']:anns[a['image_id']].append(a)
    count=Counter()
    images=coco['images'][:1] if trainer.fast_dev_run or (isinstance(trainer.limit_val_batches,int) and trainer.limit_val_batches==1) else coco['images']
    for image in images:
     frame=cv2.imread(str(self.dataset/'frames'/image['file_name']));assert frame is not None
     tensor=torch.from_numpy(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1).unsqueeze(0).float().to(pl_module.device)/255
     with torch.inference_mode():raw=adapter(tensor).cpu().tolist()
     pred=set()
     for x0,y0,x1,y1,score,cat in raw:
      if score<.25 or int(cat) not in (2,3):continue
      sid=assign_to_slot([x0,y0,x1-x0,y1-y0],self.slots,'G')
      if sid:pred.add(sid)
     present={a['slot_id'] for a in anns[image['id']]};gt={a['slot_id'] for a in anns[image['id']] if a['category_id'] in (2,3)}
     count.update(frames=1,defective_slot_frames=len(gt),good_slot_frames=len(present-gt),detected_defects=len(pred&gt),missed_defects=len(gt-pred),false_alarms_good=len((pred-gt)&present),false_alarms_empty=len(pred-present))
    result['variants'][variant]=dict(count)
   with (self.output/'paired-validation.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
  finally:pl_module.model.train(was_training)
