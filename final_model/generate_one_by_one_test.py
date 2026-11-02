"""Generate ten fresh one-vial-at-a-time synthetic videos and inspect with 006-final."""
import csv,hashlib,json,shutil,subprocess,sys,traceback
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import cv2,av
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'csi_digital_twin_v1/tools'))
from full_frame_detection import frame_annotations
from inspect_video import H264VideoWriter

def main():
 out=ROOT/'final_model/one-by-one-test-006';out.mkdir(exist_ok=False)
 for name in ('videos','sequences','raw-states','render-logs','inspection'):(out/name).mkdir()
 shutil.copy2(ROOT/'final_model/dataset-v3/calibration.json',out/'calibration.json')
 model=ROOT/'final_model/models/rtdetr-v3-006-final.ts';status=out/'status.json';inventory=[];metrics=[];totals={};last={}
 config={'videos':10,'seed_base':6100000,'fps':5,'hold_frames':5,'states_per_video':31,'expected_frames_per_video':155,'model':str(model),'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'cap_threshold':.9,'presence_threshold':.25,'subframes':16,'video_codec':'H.264 libx264 CRF18 yuv420p','controls':2,'defect_probability_other_videos':.1,'limitations':['Static placement states held for one second; no physical vial insertion, arm motion or motion blur.','New seeds and source arrangements, fixed camera and shared geometry families; synthetic development test only.']}
 (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 def state(stage,**kw):
  last.update(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),completed_videos=len(inventory),target_videos=10,**kw)
  tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(last,indent=2)+'\n');tmp.replace(status)
 try:
  for number in range(10):
   sid=f'test_one_by_one_{number:03d}';seed=config['seed_base']+number;probability=0. if number<2 else .1
   raw=out/'raw-states'/sid;raw.mkdir();state('RENDERING',sequence=sid,seed=seed,defect_probability=probability)
   command=[sys.executable,str(ROOT/'csi_digital_twin_v1/tools/isaac_headless.py'),str(ROOT/'final_model/tools/render_video_states.py'),'--camera','CapSealCam','--stage','stations/tray_inspection/tray_inspection.usdc','--randomize-trays','--variations','31','--min-vials','0','--max-vials','30','--seed',str(seed),'--defect-probability',str(probability),'--export-detections','--subframes','16','--sequence-scenario','fill','--vary-sequence','--one-by-one','--out',str(raw/'state.png')]
   with (out/'render-logs'/(sid+'.log')).open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
   sidecars=sorted(raw.glob('*.placements.json'));assert len(sidecars)==31
   state('ENCODING',sequence=sid);writer=H264VideoWriter(out/'videos'/(sid+'.mp4'),5,(640,480));frames=[];events=[];previous=set();stable={}
   try:
    for index,path in enumerate(sidecars):
     meta=json.loads(path.read_text());assert meta['sequence_state']==index
     full=frame_annotations(meta,'G');present={d['slot_id'] for d in full['slots'] if d['present']}
     assert present=={f'G{i:02d}' for i in range(1,index+1)}
     assert not previous-present
     if index:assert len(present-previous)==1
     for slot in full['slots']:
      if not slot['present']:continue
      identity=(slot['source_id'],slot.get('cap_present'),slot.get('cap_tilt_degrees'))
      if slot['slot_id'] in stable:assert stable[slot['slot_id']]==identity
      else:stable[slot['slot_id']]=identity
     previous=present
     image=cv2.imread(str(raw/meta['image']));assert image is not None and image.shape[:2]==(480,640)
     events.append({'state_index':index,'frame_index':len(frames),'timestamp_seconds':len(frames)/5,'occupied_count':index,'added_slot':f'G{index:02d}' if index else None,'gap_slots':full['gap_errors']})
     for _ in range(5):
      writer.write(image)
      frames.append({'frame_index':len(frames),'timestamp_seconds':len(frames)/5,'state_index':index,'file_name':str((raw/meta['image']).relative_to(out)),'good_tray_order':full['good_tray_order'],'gap_slots':full['gap_errors'],'annotations':full['annotations']})
   finally:writer.release()
   assert len(frames)==155
   seq={'sequence_id':sid,'split':'test','seed':seed,'scenario':'one_by_one_fill','fps':5,'frame_count':155,'video':'videos/'+sid+'.mp4','events':events,'frames':frames,'defect_plan':meta['defect_plan'],'defect_probability':probability,'limitations':config['limitations']}
   labels=out/'sequences'/(sid+'.json');labels.write_text(json.dumps(seq,indent=2)+'\n')
   state('INSPECTING',sequence=sid)
   inspected=out/'inspection'/sid
   command=[sys.executable,'-s',str(ROOT/'final_model/inspect_video.py'),'--video',str(out/'videos'/(sid+'.mp4')),'--labels',str(labels),'--calibration',str(out/'calibration.json'),'--model',str(model),'--threshold','.90','--presence-threshold','.25','--output',str(inspected)]
   with (out/'render-logs'/(sid+'-inspection.log')).open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
   predictions=[json.loads(line) for line in (inspected/'frame-results.jsonl').read_text().splitlines()];assert len(predictions)==155
   counts={mode:Counter() for mode in ('raw_cap','confirmed_cap','settled_cap','missing_vial','order','settled_order')}
   for truth,pred in zip(frames,predictions):
    assert pred['ground_truth_available'];assert pred['frame_index']==truth['frame_index']
    gt={d['slot_id'] for d in truth['annotations'] if d['category_id'] in (2,3)};present={d['slot_id'] for d in truth['annotations']}
    assert set(pred['actual_defects']['cap_defects'])==gt
    settled=truth['frame_index']%5>=2
    modes={'raw_cap':{d['slot_id'] for d in pred['raw_cap_defects']},'confirmed_cap':{d['slot_id'] for d in pred['cap_defects']}}
    if settled:modes['settled_cap']=modes['confirmed_cap']
    for mode,detected in modes.items():counts[mode].update(frames=1,defective_slot_frames=len(gt),detected_defects=len(gt&detected),missed_defects=len(gt-detected),good_slot_frames=len(present-gt),false_alarms_good=len((detected-gt)&present),false_alarms_empty=len(detected-present))
    counts['missing_vial'].update(frames=1,false_missing_slot_frames=len(pred['missing_slots']))
    actual=len(present)==30;found=pred['order_status']=='complete'
    c=Counter(frames=1,actual_complete_frames=int(actual),correct_complete_frames=int(actual and found),missed_complete_frames=int(actual and not found),false_complete_frames=int(found and not actual))
    counts['order'].update(c)
    if settled:counts['settled_order'].update(c)
   for video in (out/'videos'/(sid+'.mp4'),inspected/'inspection.mp4'):
    with av.open(str(video)) as container:
     assert container.streams.video[0].codec_context.name=='h264';assert sum(1 for _ in container.decode(video=0))==155
   cap=cv2.VideoCapture(str(inspected/'inspection.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,154);ok,last_frame=cap.read();cap.release();assert ok;cv2.imwrite(str(inspected/'last-frame.png'),last_frame)
   row={'sequence_id':sid,'seed':seed,'control':number<2,'defect_plan':meta['defect_plan'],'source_video':'videos/'+sid+'.mp4','inspected_video':'inspection/'+sid+'/inspection.mp4','metrics':{k:dict(v) for k,v in counts.items()}};metrics.append(row)
   for mode,c in counts.items():totals.setdefault(mode,Counter()).update(c)
   inventory.append(sid)
   (out/'results.json').write_text(json.dumps({'config':config,'completed_videos':len(inventory),'per_video':metrics,'aggregate':totals},indent=2)+'\n')
   lines=['# One-vial-at-a-time test videos','','006-final; cap threshold 0.90, presence 0.25. Each source video starts empty and adds one vial per second, G01 through G30. Source and annotated outputs are H.264. Annotated outputs show independent Actual / Model / Confirmed summaries.','','| Video | Type | Source | Annotated inspection |','|---|---|---|---|']
   for r in metrics:lines.append(f"| {r['sequence_id']} | {'Clean control' if r['control'] else '10% per-vial defect sampling'} | [Source]({r['source_video']}) | [Inspection]({r['inspected_video']}) |")
   (out/'video-index.md').write_text('\n'.join(lines)+'\n');state('VIDEO_COMPLETE',sequence=sid);print('COMPLETE',sid,flush=True)
  state('COMPLETE',frames=1550,rendered_states=310);print('ALL COMPLETE',flush=True)
 except BaseException:state('ERROR',traceback=traceback.format_exc());raise
if __name__=='__main__':main()
