"""Measure warmed serial batch-one TorchScript and full video inspection latency."""
import csv,hashlib,json,sys,time,platform
from pathlib import Path
from datetime import datetime,timezone
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'csi_digital_twin_v1/tools'))
from full_frame_detection import assign_to_slot
from temporal import VideoInspector

def stats(values):
 a=np.array(values)*1000
 return {'samples':len(a),'mean_ms':float(a.mean()),'median_ms':float(np.median(a)),'p95_ms':float(np.percentile(a,95)),'p99_ms':float(np.percentile(a,99)),'min_ms':float(a.min()),'max_ms':float(a.max()),'fps_from_mean':float(1000/a.mean())}
def tensor(frame):return torch.from_numpy(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1).float().unsqueeze(0).cuda()/255

def main():
 out=ROOT/'final_model/inference-speed-006';assert not (out/'results.json').exists()
 modelpath=ROOT/'final_model/models/rtdetr-v3-006-final.ts';meta=json.loads(modelpath.with_suffix('.json').read_text())
 for lib in meta['required_libraries']:
  path=modelpath.parent/lib['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==lib['sha256'];torch.ops.load_library(str(path))
 start=time.perf_counter();model=torch.jit.load(str(modelpath),map_location='cuda').eval();torch.cuda.synchronize();load_s=time.perf_counter()-start
 videos=sorted((ROOT/'final_model/dataset-v3/videos').glob('test_*.mp4'));assert len(videos)==15
 cap=cv2.VideoCapture(str(videos[0]));ok,frame=cap.read();cap.release();assert ok
 sample=tensor(frame);slots=json.loads((ROOT/'final_model/dataset-v3/calibration.json').read_text())['slots']
 times=[];stage={k:[] for k in ('decode','prepare_and_transfer','forward_and_copy','slot_temporal','draw_write','total')}
 with torch.inference_mode(),torch.jit.optimized_execution(False):
  start=time.perf_counter()
  for _ in range(30):model(sample)
  torch.cuda.synchronize();warmup_s=time.perf_counter()-start
  torch.cuda.reset_peak_memory_stats()
  for _ in range(300):
   torch.cuda.synchronize();start=time.perf_counter();prediction=model(sample);torch.cuda.synchronize();times.append(time.perf_counter()-start)
  del prediction
  print('Model-only complete',stats(times),flush=True)
  rows=[];pipeline_start=time.perf_counter()
  with (out/'frame-results.jsonl').open('w') as results:
   for video in videos:
    cap=cv2.VideoCapture(str(video));assert cap.isOpened();fps=cap.get(cv2.CAP_PROP_FPS);tracker=VideoInspector(slots,3,3,.25)
    writer=cv2.VideoWriter(str(out/(video.stem+'-inspection.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),fps,(640,480));assert writer.isOpened();index=0
    while True:
     a=time.perf_counter();ok,frame=cap.read();b=time.perf_counter()
     if not ok:break
     sample=tensor(frame);torch.cuda.synchronize();c=time.perf_counter()
     raw=model(sample).detach().cpu().numpy();torch.cuda.synchronize();d=time.perf_counter()
     assigned=[]
     for x0,y0,x1,y1,score,cat in raw:
      bbox=[float(x0),float(y0),float(x1-x0),float(y1-y0)];sid=assign_to_slot(bbox,slots,'G')
      if sid:assigned.append({'slot_id':sid,'bbox':bbox,'score':float(score),'category_id':int(cat)})
     status=tracker.update(assigned,index,index/fps);e=time.perf_counter()
     results.write(json.dumps(dict(video=video.stem,**status))+'\n')
     cv2.putText(frame,f"G: {status['occupied_count']}/30 {status['order_status']}",(12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0) if status['order_status']=='complete' else (0,180,255),2)
     for det in assigned:
      if det['category_id']==1 or det['score']<.25:continue
      x,y,w,h=map(round,det['bbox']);cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2);cv2.putText(frame,f"{det['slot_id']} {det['score']:.2f}",(x,y),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
     writer.write(frame);f=time.perf_counter()
     values=[b-a,c-b,d-c,e-d,f-e,f-a]
     for k,value in zip(stage,values):stage[k].append(value)
     rows.append(dict(video=video.stem,frame=index,**{k:v*1000 for k,v in zip(stage,values)}));index+=1
    cap.release();writer.release();assert index==32
    print('pipeline',video.stem,index,flush=True)
  pipeline_wall=time.perf_counter()-pipeline_start
 report={'device':torch.cuda.get_device_name(0),'compute_capability':torch.cuda.get_device_capability(),'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'created_utc':datetime.now(timezone.utc).isoformat(),'model':str(modelpath),'model_sha256':hashlib.sha256(modelpath.read_bytes()).hexdigest(),'batch_size':1,'native_frame_size':[640,480],'network_size':[640,640],'precision':'FP32','threshold':.25,'confirmation_frames':3,'model_load_seconds':load_s,'warmup_iterations':30,'warmup_seconds':warmup_s,'model_only':stats(times),'pipeline':{k:stats(v) for k,v in stage.items()},'pipeline_wall_seconds':pipeline_wall,'pipeline_frames':len(rows),'pipeline_fps_including_video_open_close':len(rows)/pipeline_wall,'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),'notes':['Model-only repeats one preloaded GPU frame, includes exported resize/preprocessing/postprocessing, excludes decoding, input transfer, CPU output copy, slot mapping and recording. GPU is synchronized for each timing.','Full pipeline processes 480 frames from 15 held-out MP4s as fast as possible, including decode, transfer, inference, CPU output copy, slot mapping, temporal fusion, overlay, JSONL writing and MP4 encoding. Explicit stage synchronization prevents GPU work leaking into later timings.','Reported times exclude application startup/model load and GPU warmup; those are listed separately.','No camera/network capture or display latency measured. GPU throughput on prerecorded clips is not a guarantee of sustained camera performance.']}
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 with (out/'frame-timings.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
