#!/usr/bin/env python3
"""Decode full video, run a TorchScript detector or replay external predictions, fuse G slots."""
import argparse,json,sys,hashlib
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from full_frame_detection import assign_to_slot
from temporal import VideoInspector

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model',type=Path,help='TorchScript detector: float RGB BCHW in [0,1], output Nx6 xyxy/score/category in input pixels')
    mode.add_argument('--predictions',type=Path,help='External detector JSON records: frame_index, bbox xywh, category_id, score')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--confirm',type=int,default=3)
    p.add_argument('--threshold',type=float,default=.1);p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.output.exists():p.error('Output exists')
    calibration=json.loads(args.calibration.read_text());slots=calibration['slots']
    tracker=VideoInspector(slots,args.confirm,args.confirm,args.threshold)
    predictions=defaultdict(list)
    if args.predictions:
        for d in json.loads(args.predictions.read_text()):
            if not isinstance(d['frame_index'],int) or d['frame_index']<0:raise ValueError('Prediction frame_index must be nonnegative integer')
            predictions[d['frame_index']].append(d)
    else:
        import torch
        metadata_path=args.model.with_suffix('.json')
        if metadata_path.exists():
            metadata=json.loads(metadata_path.read_text())
            if metadata.get('supported_device')=='cuda' and not args.device.startswith('cuda'):
                raise ValueError('This exported detector requires CUDA')
            for library in metadata.get('required_libraries',[]):
                lib=(args.model.parent/library['file']).resolve()
                if lib.parent!=args.model.parent.resolve():raise ValueError('Operator library must be next to model')
                if hashlib.sha256(lib.read_bytes()).hexdigest()!=library['sha256']:raise ValueError('Operator library checksum mismatch')
                torch.ops.load_library(str(lib))
        model=torch.jit.load(str(args.model),map_location=args.device).eval()
    capture=cv2.VideoCapture(str(args.video))
    if not capture.isOpened():raise ValueError('Cannot open video')
    fps=capture.get(cv2.CAP_PROP_FPS)
    if fps<=0:raise ValueError('Video has no valid FPS')
    args.output.mkdir(parents=True);writer=None;count=0;complete=0
    try:
        with (args.output/'frame-results.jsonl').open('w') as results:
            while True:
                ok,frame=capture.read()
                if not ok:break
                height,width=frame.shape[:2]
                if [width,height]!=calibration['image_size']:raise ValueError('Video and calibration coordinates differ')
                if args.predictions:detections=predictions.get(count,[])
                else:
                    tensor=torch.from_numpy(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1).float().unsqueeze(0).to(args.device)/255
                    with torch.inference_mode(), torch.jit.optimized_execution(False):raw=model(tensor).detach().cpu().numpy()
                    if raw.ndim==3:raw=raw[0]
                    if raw.ndim!=2 or raw.shape[1]!=6:raise ValueError('Expected Nx6 detector output')
                    detections=[{'bbox':[float(x0),float(y0),float(x1-x0),float(y1-y0)],'score':float(score),'category_id':int(cat)} for x0,y0,x1,y1,score,cat in raw]
                assigned=[]
                for detection in detections:
                    # Never trust a provided slot ID; map native full-frame coordinates.
                    slot=assign_to_slot(detection['bbox'],slots,'G')
                    if slot:assigned.append(dict(detection,slot_id=slot))
                status=tracker.update(assigned,count,count/fps);results.write(json.dumps(status)+'\n')
                complete+=status['order_status']=='complete'
                cv2.putText(frame,f"G: {status['occupied_count']}/30 {status['order_status']}",(12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0) if status['order_status']=='complete' else (0,180,255),2)
                for d in assigned:
                    if d['category_id']==1 or d['score']<args.threshold:continue
                    x,y,w,h=map(round,d['bbox']);cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                    cv2.putText(frame,f"{d['slot_id']}\n{d['score']:.{2}f}",(x,y),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
                if writer is None:
                    writer=cv2.VideoWriter(str(args.output/'inspection.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
                    if not writer.isOpened():raise RuntimeError('Cannot open video encoder')
                writer.write(frame);count+=1
    finally:
        capture.release()
        if writer:writer.release()
    if not count:raise ValueError('Video contained no decoded frames')
    if args.predictions and any(index>=count for index in predictions):raise ValueError('Predictions reference frames outside the video')
    (args.output/'summary.json').write_text(json.dumps({'frames':count,'fps':fps,'complete_frames':complete,
        'mode':'external_prediction_replay' if args.predictions else 'trained_detector',
        'model':str(args.model) if args.model else None,'threshold':args.threshold,'confirmation_frames':args.confirm,
        'note':'External prediction replay; no accuracy score is computed.' if args.predictions else 'Learned detector used; complete-frame counts are outputs, not ground-truth accuracy scores.'},indent=2)+'\n')
if __name__=='__main__':main()
