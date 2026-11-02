"""Fixed-threshold evaluation of exported detector across current and legacy test videos."""
import argparse,csv,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'csi_digital_twin_v1/tools'))
from full_frame_detection import assign_to_slot
from temporal import VideoInspector

def writecsv(path,rows):
 if not rows:return
 with path.open('w',newline='') as stream:
  w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--threshold',type=float,default=.9);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if not 0<=a.threshold<=1:p.error('Invalid cutoff')
 a.output.mkdir(parents=True,exist_ok=False);out=a.output.resolve();(out/'videos').mkdir()
 modelpath=ROOT/'final_model/models/rtdetr-v3-006-final.ts';meta=json.loads(modelpath.with_suffix('.json').read_text())
 for lib in meta['required_libraries']:
  path=modelpath.parent/lib['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==lib['sha256'];torch.ops.load_library(str(path))
 model=torch.jit.load(str(modelpath),map_location='cuda').eval()
 thresholds=sorted(set([a.threshold,.25]));totals={};video_rows=[];frame_rows=[];slot_rows=[];inventory=[];seen_hashes=set()
 with torch.inference_mode(),torch.jit.optimized_execution(False),(out/'frame-results.jsonl').open('w') as details:
  for dataset_name in ('dataset-v3','dataset-v2','dataset'):
   data=ROOT/'final_model'/dataset_name;slots=json.loads((data/'calibration.json').read_text())['slots']
   group={str(t):{mode:Counter() for mode in ('raw','confirmed','confirmed_settled')} for t in thresholds};videos=0
   for videopath in sorted((data/'videos').glob('test_*.mp4')):
    sha=hashlib.sha256(videopath.read_bytes()).hexdigest();assert sha not in seen_hashes,'Duplicate video content';seen_hashes.add(sha)
    seq=json.loads((data/'sequences'/videopath.with_suffix('.json').name).read_text());assert seq['split']=='test'
    cap=cv2.VideoCapture(str(videopath));assert cap.isOpened();fps=cap.get(cv2.CAP_PROP_FPS);assert fps>0
    target=out/'videos'/f'{dataset_name}__{videopath.name}';writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,(640,480));assert writer.isOpened()
    trackers={t:VideoInspector(slots,3,3,t) for t in thresholds};local={str(t):{mode:Counter() for mode in ('raw','confirmed','confirmed_settled')} for t in thresholds}
    histories=defaultdict(lambda:dict(present_frames=0,defective_frames=0,raw_detected_frames=0,confirmed_frames=0,raw_max_score=0.,ever_confirmed=False,confirmed_then_cleared_frames=0))
    index=0
    while True:
     ok,frame=cap.read()
     if not ok:break
     truth=seq['frames'][index];assert truth['frame_index']==index and frame.shape[:2]==(480,640)
     present={ann['slot_id'] for ann in truth['annotations']};gt={ann['slot_id'] for ann in truth['annotations'] if ann['category_id'] in (2,3)}
     tensor=torch.from_numpy(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1).float().unsqueeze(0).cuda()/255
     raw=model(tensor).cpu().tolist();assigned=[]
     for x0,y0,x1,y1,score,cat in raw:
      box=[float(x0),float(y0),float(x1-x0),float(y1-y0)];sid=assign_to_slot(box,slots,'G')
      if sid:assigned.append({'slot_id':sid,'bbox':box,'score':float(score),'category_id':int(cat)})
     settled=index>=2 and seq['frames'][index-2]['state_index']==truth['state_index']
     for t,tracker in trackers.items():
      status=tracker.update(assigned,index,index/fps)
      pred={d['slot_id'] for d in assigned if d['score']>=t and d['category_id'] in (2,3)};confirmed={d['slot_id'] for d in status['defects']}
      modes={'raw':pred,'confirmed':confirmed}
      if settled:modes['confirmed_settled']=confirmed
      for mode,detected in modes.items():
       c=Counter(frames=1,defective_slot_frames=len(gt),good_slot_frames=len(present-gt),detected_defects=len(gt&detected),missed_defects=len(gt-detected),false_alarms_good=len((detected-gt)&present),false_alarms_empty=len(detected-present))
       if mode!='raw':
        actual_complete=len(present)==30;predicted_complete=status['order_status']=='complete'
        c.update(complete_truth_frames=int(actual_complete),complete_predicted_frames=int(predicted_complete),false_complete_frames=int(predicted_complete and not actual_complete),missed_complete_frames=int(actual_complete and not predicted_complete))
       group[str(t)][mode].update(c);local[str(t)][mode].update(c)
       frame_rows.append(dict(dataset=dataset_name,video=videopath.stem,frame=index,threshold=t,mode=mode,**{k:c.get(k,0) for k in ('frames','defective_slot_frames','good_slot_frames','detected_defects','missed_defects','false_alarms_good','false_alarms_empty','complete_truth_frames','complete_predicted_frames','false_complete_frames','missed_complete_frames')}))
      if t==a.threshold:
       scores={sid:max((d['score'] for d in assigned if d['slot_id']==sid and d['category_id'] in (2,3)),default=0.) for sid in present|pred|confirmed}
       for sid in present|pred|confirmed:
        h=histories[sid];h['present_frames']+=sid in present;h['defective_frames']+=sid in gt;h['raw_detected_frames']+=sid in pred;h['confirmed_frames']+=sid in confirmed;h['raw_max_score']=max(h['raw_max_score'],scores[sid])
        if sid in gt and h['ever_confirmed'] and sid not in confirmed:h['confirmed_then_cleared_frames']+=1
        h['ever_confirmed']|=sid in confirmed
       details.write(json.dumps(dict(dataset=dataset_name,video=videopath.stem,ground_truth_defects=sorted(gt),raw_defect_slots=sorted(pred),defect_scores=scores,**status))+'\n')
       display=status;display_pred=pred
     cv2.putText(frame,f'{dataset_name}/{videopath.stem} cutoff={a.threshold:.2f}',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
     cv2.putText(frame,f"Predicted G {display['occupied_count']}/30 {display['order_status']}",(8,39),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,220,255),1)
     cv2.putText(frame,'Truth defects: '+(','.join(sorted(gt)) or 'none'),(8,459),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,200,0),1)
     cv2.putText(frame,'Confirmed: '+(','.join(sorted({d['slot_id'] for d in display['defects']})) or 'none'),(8,477),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
     for d in assigned:
      if d['category_id'] not in (2,3) or d['score']<a.threshold:continue
      x,y,w,h=map(round,d['bbox']);cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2);cv2.putText(frame,f"{d['slot_id']} {d['score']:.2f}",(x,y),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
     writer.write(frame);index+=1
    cap.release();writer.release();assert index==seq['frame_count']
    check=cv2.VideoCapture(str(target));decoded=0
    while check.read()[0]:decoded+=1
    check.release();assert decoded==index,'Saved video failed verification'
    for t,modes in local.items():
     for mode,c in modes.items():video_rows.append(dict(dataset=dataset_name,video=videopath.stem,threshold=t,mode=mode,**{k:c.get(k,0) for k in ('frames','defective_slot_frames','good_slot_frames','detected_defects','missed_defects','false_alarms_good','false_alarms_empty','complete_truth_frames','complete_predicted_frames','false_complete_frames','missed_complete_frames')}))
    for sid,h in sorted(histories.items()):slot_rows.append(dict(dataset=dataset_name,video=videopath.stem,slot_id=sid,**h))
    inventory.append({'dataset':dataset_name,'video':videopath.stem,'seed':seq.get('seed'),'scenario':seq.get('scenario'),'sha256':sha,'frames':index,'saved_video':str(target.relative_to(out))});videos+=1;print(dataset_name,videopath.stem,index,flush=True)
   totals[dataset_name]={'videos':videos,'thresholds':group}
 report={'model':str(modelpath),'model_sha256':hashlib.sha256(modelpath.read_bytes()).hexdigest(),'threshold':a.threshold,'reference_threshold':.25,'videos':len(inventory),'frames':sum(v['frames'] for v in inventory),'groups':totals,'inventory':inventory,'notes':['dataset-v3 is the main benchmark; dataset-v2 and dataset are supplementary older synthetic test scenarios.','No training or threshold selection performed. These test videos have been examined previously and are a development benchmark, not a fresh generalization test.','Raw red boxes are individual-frame candidates; confirmed uses three-frame logic. Settled omits first two frames after each placement change.','Any defect subtype counts as a defect alarm; counts are correlated slot-frames, not unique independent vials.','Same cutoff applies to vial presence and defect categories, matching inspect_video.py; high cutoff can impair occupancy and order-completeness output.','Annotated videos are re-encoded only after inference; scores use the original decoded input frames.']}
 (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');writecsv(out/'video-metrics.csv',video_rows);writecsv(out/'frame-metrics.csv',frame_rows);writecsv(out/'slot-histories.csv',slot_rows);print('COMPLETE',flush=True)
if __name__=='__main__':main()
