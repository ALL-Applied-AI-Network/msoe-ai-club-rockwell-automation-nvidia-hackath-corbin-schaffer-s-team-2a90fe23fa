"""Compare raw slot-level defect alarms against validation labels without changing model thresholds."""
import csv,json,sys,math
from pathlib import Path
from collections import Counter,defaultdict
import torch
from PIL import Image,ImageDraw
import numpy as np
from omegaconf import OmegaConf
from detector_adapter import RTDETRVideoAdapter
from nvidia_tao_pytorch.cv.rtdetr.model.pl_rtdetr_model import RTDETRPlModel
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'csi_digital_twin_v1/tools'))
from full_frame_detection import assign_to_slot

def main():
 out=ROOT/'final_model/false-alarm-audit';out.mkdir(exist_ok=True)
 dataset=ROOT/'final_model/dataset-v3';cal=json.loads((dataset/'calibration.json').read_text())['slots']
 coco=json.loads((dataset/'annotations/validation-keyframes.json').read_text());annotations=defaultdict(list)
 for ann in coco['annotations']:annotations[ann['image_id']].append(ann)
 summary={};details=[];bad_images=[]
 for name,checkpoint in [('initial','training-v3-001/model_best_026.pth'),('continued','training-v3-002/model_best_032.pth')]:
  cfg=OmegaConf.load(ROOT/'final_model/training-v3-001/spec.yaml');cfg.model.pretrained_backbone_path=''
  model=RTDETRPlModel(cfg)
  saved=torch.load(ROOT/'final_model'/checkpoint,map_location='cpu',weights_only=False)
  model.load_state_dict(saved['state_dict'],strict=True);del saved
  adapter=RTDETRVideoAdapter(model.model.cuda().eval(),tuple(cfg.dataset.augmentation.eval_spatial_size),int(cfg.model.num_select)).cuda().eval()
  thresholds=[.1,.25,.5,.75];counts={str(t):Counter() for t in thresholds}
  for im in coco['images']:
   with Image.open(dataset/'frames'/im['file_name']) as image:array=np.asarray(image.convert('RGB')).copy()
   tensor=torch.from_numpy(array).permute(2,0,1).unsqueeze(0).float().cuda()/255
   with torch.inference_mode():raw=adapter(tensor).cpu().tolist()
   truth=defaultdict(set);present=set()
   for ann in annotations[im['id']]:
    present.add(ann['slot_id'])
    if ann['category_id']!=1:truth[ann['slot_id']].add(ann['category_id'])
   assigned=[]
   for x0,y0,x1,y1,score,cat in raw:
    cat=int(cat)
    if cat==1 or score<.1:continue
    sid=assign_to_slot([x0,y0,x1-x0,y1-y0],cal,'G')
    if sid is None:continue
    assigned.append((sid,cat,score))
    correct=cat in truth[sid]
    row={'model':name,'frame':im['file_name'],'slot_id':sid,'category_id':cat,'score':score,'slot_present':sid in present,'ground_truth_defects':','.join(map(str,sorted(truth[sid]))),'matches_slot_class':correct}
    details.append(row)
    if not correct:bad_images.append((score,row,(x0,y0,x1,y1)))
   for threshold in thresholds:
    c=counts[str(threshold)];pred={sid for sid,cat,score in assigned if score>=threshold}
    gt={sid for sid,cats in truth.items() if cats}
    c['correct_defect_slot_frames']+=len(pred&gt);c['missed_defect_slot_frames']+=len(gt-pred)
    c['false_alarm_on_good_vial_slot_frames']+=len((pred-gt)&present)
    c['false_alarm_on_empty_slot_frames']+=len(pred-present)
    c['good_vial_slot_frames']+=len(present-gt);c['defective_slot_frames']+=len(gt)
  summary[name]={'checkpoint':checkpoint,'validation_keyframes':len(coco['images']),'thresholds':{k:dict(v) for k,v in counts.items()}}
  del adapter,model;torch.cuda.empty_cache()
 with (out/'detections.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 # Largest false positives first, with full scene context and slot identity.
 chosen=sorted(bad_images,key=lambda x:x[0],reverse=True)[:12]
 sheet=Image.new('RGB',(1280,510*math.ceil(len(chosen)/2)),'white');draw=ImageDraw.Draw(sheet)
 for i,(score,row,box) in enumerate(chosen):
  with Image.open(dataset/'frames'/row['frame']) as im:frame=im.convert('RGB')
  d=ImageDraw.Draw(frame);d.rectangle(box,outline='red',width=2)
  x=(i%2)*640;y=(i//2)*510;sheet.paste(frame,(x,y));draw.text((x+4,y+481),f"{row['model']} {row['slot_id']} class={row['category_id']} score={score:.3f} present={row['slot_present']}",fill='black')
 if chosen:sheet.save(out/'false-alarm-examples.jpg')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
