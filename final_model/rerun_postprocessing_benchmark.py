"""Run the exact prior 21-video inventory through the updated inspector."""
import argparse,csv,hashlib,json,os,subprocess,sys
from collections import Counter
from pathlib import Path
import av
ROOT=Path(__file__).resolve().parents[1]

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'final_model/threshold-090-postprocessing-006')
 args=parser.parse_args()
 previous=json.loads((ROOT/'final_model/threshold-090-evaluation-006/summary.json').read_text())
 out=args.output.resolve();out.mkdir(exist_ok=False);(out/'videos').mkdir()
 model=ROOT/'final_model/models/rtdetr-v3-006-final.ts'
 assert hashlib.sha256(model.read_bytes()).hexdigest()==previous['model_sha256']
 groups={};metrics=[];inventory=[]
 for entry in previous['inventory']:
  name=entry['dataset'];stem=entry['video'];data=ROOT/'final_model'/name;video=data/'videos'/(stem+'.mp4')
  assert hashlib.sha256(video.read_bytes()).hexdigest()==entry['sha256']
  target=out/'videos'/(name+'__'+stem)
  command=[sys.executable,'-s',str(ROOT/'final_model/inspect_video.py'),'--video',str(video),'--calibration',str(data/'calibration.json'),'--model',str(model),'--labels',str(data/'sequences'/(stem+'.json')),'--threshold','.90','--presence-threshold','.25','--confirm','3','--absence-frames','3','--output',str(target)]
  env=os.environ.copy();env['PYTHONNOUSERSITE']='1'
  with (out/(name+'__'+stem+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
  seq=json.loads((data/'sequences'/(stem+'.json')).read_text());pred=[json.loads(line) for line in (target/'frame-results.jsonl').read_text().splitlines()];assert len(pred)==entry['frames']
  totals={k:Counter() for k in ('raw_cap','confirmed_cap','settled_cap','missing_vial','settled_missing_vial','order','settled_order')}
  for i,(truth,result) in enumerate(zip(seq['frames'],pred)):
   assert result['frame_index']==truth['frame_index']==i
   present={d['slot_id'] for d in truth['annotations']};cap_gt={d['slot_id'] for d in truth['annotations'] if d['category_id'] in (2,3)}
   gap_gt=set(truth['gap_slots'])
   assert result['ground_truth_available']
   assert set(result['actual_defects']['cap_defects'])==cap_gt
   assert set(result['actual_defects']['missing_vials'])==gap_gt
   settled=i>=2 and seq['frames'][i-2]['state_index']==truth['state_index']
   modes={'raw_cap':({d['slot_id'] for d in result['raw_cap_defects']},cap_gt),'confirmed_cap':({d['slot_id'] for d in result['cap_defects']},cap_gt),'missing_vial':(set(result['missing_slots']),gap_gt)}
   if settled:modes['settled_cap']=modes['confirmed_cap'];modes['settled_missing_vial']=modes['missing_vial']
   for mode,(found,gt) in modes.items():
    totals[mode].update(frames=1,positive_slot_frames=len(gt),detected_slot_frames=len(found&gt),missed_slot_frames=len(gt-found),false_alarm_slot_frames=len(found-gt),false_alarms_on_present_slots=len((found-gt)&present),false_alarms_on_empty_slots=len(found-present-gt))
   actual=len(present)==30;detected=result['order_status']=='complete'
   c=Counter(frames=1,actual_complete_frames=int(actual),predicted_complete_frames=int(detected),correct_complete_frames=int(actual and detected),missed_complete_frames=int(actual and not detected),false_complete_frames=int(detected and not actual))
   totals['order'].update(c)
   if settled:totals['settled_order'].update(c)
  with av.open(str(target/'inspection.mp4')) as container:
   assert container.streams.video[0].codec_context.name=='h264';assert sum(1 for _ in container.decode(video=0))==entry['frames']
  for mode,c in totals.items():groups.setdefault(name,{}).setdefault(mode,Counter()).update(c)
  row={'dataset':name,'video':stem,'frames':len(pred),'results':{mode:dict(c) for mode,c in totals.items()},'saved_video':str((target/'inspection.mp4').relative_to(out))};metrics.append(row)
  inventory.append(entry)
  print('COMPLETE',name,stem,'missing',dict(totals['missing_vial']),flush=True)
 aggregate={}
 for modes in groups.values():
  for mode,c in modes.items():aggregate.setdefault(mode,Counter()).update(c)
 report={'model':str(model),'model_sha256':previous['model_sha256'],'videos':len(inventory),'frames':sum(e['frames'] for e in inventory),'cap_defect_threshold':.9,'presence_threshold':.25,'confirm_frames':3,'absence_frames':3,'input_inventory_matches_previous':True,'ground_truth_summary_verified_every_frame':True,'previous_run':'final_model/threshold-090-evaluation-006','groups':groups,'aggregate':aggregate,'per_video':metrics,'inventory':inventory,'notes':['Model weights, input video hashes and cap cutoff match the prior 21-video benchmark. Presence cutoff is now 0.25 instead of the prior shared 0.90.','Missing-vial postprocessing and H.264 red/green/yellow overlays are enabled.','Cap defects, missing-vial alerts and tray completeness are scored separately.','Missing-vial ground truth is existing fill-order gap labels; these clips do not test removal of previously occupied vials.','Settled metrics omit first two frames after every placement-state change. Counts are correlated slot-frames, not independent vials.']}
 (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 with (out/'video-metrics.csv').open('w',newline='') as stream:
  rows=[dict(dataset=r['dataset'],video=r['video'],mode=mode,**{k:c.get(k,0) for k in ('frames','positive_slot_frames','detected_slot_frames','missed_slot_frames','false_alarm_slot_frames','actual_complete_frames','predicted_complete_frames','correct_complete_frames','missed_complete_frames','false_complete_frames')}) for r in metrics for mode,c in r['results'].items()]
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print('ALL COMPLETE',flush=True)
if __name__=='__main__':main()
