"""Fixed-threshold comparison on all held-out decoded test videos."""
import csv,json,sys
from pathlib import Path
from collections import Counter
import cv2,numpy as np,torch
from omegaconf import OmegaConf
from detector_adapter import RTDETRVideoAdapter
from temporal import VideoInspector
from nvidia_tao_pytorch.cv.rtdetr.model.pl_rtdetr_model import RTDETRPlModel
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'csi_digital_twin_v1/tools'))
from full_frame_detection import assign_to_slot

def main():
 out=ROOT/'final_model/overfitting-comparison-005';out.mkdir(exist_ok=True);assert not (out/'summary.json').exists()
 data=ROOT/'final_model/dataset-v3';slots=json.loads((data/'calibration.json').read_text())['slots']
 sequences=[json.loads(p.read_text()) for p in sorted((data/'sequences').glob('test_*.json')) if '.oracle-predictions.' not in p.name]
 assert len(sequences)==15
 results={};rows=[]
 for name,filename in [('best','model_best_032.pth'),('final','final_epoch_327.pth')]:
  checkpoint=ROOT/'final_model/training-v3-005'/filename
  cfg=OmegaConf.load(checkpoint.parent/'spec.yaml');cfg.model.pretrained_backbone_path=''
  model=RTDETRPlModel(cfg);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
  model.load_state_dict(saved['state_dict'],strict=True);del saved
  adapter=RTDETRVideoAdapter(model.model.cuda().eval(),tuple(cfg.dataset.augmentation.eval_spatial_size),int(cfg.model.num_select)).cuda().eval()
  totals={str(t):{mode:Counter() for mode in ('raw','confirmed','confirmed_settled')} for t in (.1,.25,.5,.75)}
  for seq in sequences:
   trackers={t:VideoInspector(slots,3,3,t) for t in (.1,.25,.5,.75)}
   cap=cv2.VideoCapture(str(data/seq['video']));index=0
   while True:
    ok,frame=cap.read()
    if not ok:break
    truth_frame=seq['frames'][index];assert truth_frame['frame_index']==index
    present={a['slot_id'] for a in truth_frame['annotations']}
    gt={a['slot_id'] for a in truth_frame['annotations'] if a['category_id'] in (2,3)}
    tensor=torch.from_numpy(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1).unsqueeze(0).float().cuda()/255
    with torch.inference_mode():raw=adapter(tensor).cpu().tolist()
    assigned=[]
    for x0,y0,x1,y1,score,cat in raw:
     if score<.1:continue
     box=[x0,y0,x1-x0,y1-y0];sid=assign_to_slot(box,slots,'G')
     if sid:assigned.append({'slot_id':sid,'bbox':box,'category_id':int(cat),'score':score})
    # Exclude first two frames after a placement change for settled temporal metrics.
    settled=index>=2 and seq['frames'][index-2]['state_index']==truth_frame['state_index']
    for t,tracker in trackers.items():
     status=tracker.update(assigned,index,index/seq['fps'])
     preds={'raw':{d['slot_id'] for d in assigned if d['score']>=t and d['category_id'] in (2,3)},'confirmed':{d['slot_id'] for d in status['defects']}}
     if settled:preds['confirmed_settled']=preds['confirmed']
     for mode,pred in preds.items():
      c=Counter(frames=1,defective_slot_frames=len(gt),good_vial_slot_frames=len(present-gt),true_positives=len(gt&pred),missed_defects=len(gt-pred),false_alarms_good=len((pred-gt)&present),false_alarms_empty=len(pred-present))
      totals[str(t)][mode].update(c)
      rows.append(dict(model=name,video=seq['sequence_id'],frame_index=index,threshold=t,mode=mode,**c))
    index+=1
   cap.release();assert index==seq['frame_count']
   print(name,seq['sequence_id'],index,flush=True)
  results[name]={'checkpoint':str(checkpoint),'thresholds':totals}
  del adapter,model;torch.cuda.empty_cache()
 report={'test_videos':len(sequences),'frames_per_model':sum(s['frame_count'] for s in sequences),'notes':['All decoded MP4 frames, identical preprocessing and fixed thresholds; no test-set tuning.','Counts are slot-frames, not independent vials; held frames are correlated.','Any defect category counts as an alarm; this measures defect presence rather than correct defect subtype.','Confirmed uses the existing three-frame temporal logic; confirmed_settled excludes first two frames after each placement change.'],'models':results}
 (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 with (out/'frame-metrics.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
