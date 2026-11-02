#!/usr/bin/env python3
"""Decode full video, run a detector or replay labeled detections, and fuse G slots."""
import argparse,json,sys,hashlib
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'csi_digital_twin_v1/tools'))
from full_frame_detection import assign_to_slot
from temporal import VideoInspector


class H264VideoWriter:
    """Write browser-compatible H.264/yuv420p MP4, including delayed frames."""
    def __init__(self,path,fps,size):
        try:
            import av
        except ImportError as exc:
            raise RuntimeError('H.264 output requires PyAV: install it with python -m pip install av') from exc
        from fractions import Fraction
        width,height=size
        if width%2 or height%2:
            raise ValueError('H.264 yuv420p output requires even frame dimensions')
        self.av=av
        self.container=av.open(str(path),mode='w',format='mp4',options={'movflags':'+faststart'})
        try:
            self.stream=self.container.add_stream('libx264',rate=Fraction(str(fps)).limit_denominator(100000),
                                                  options={'crf':'18','preset':'fast'})
            self.stream.width=width;self.stream.height=height
            self.stream.pix_fmt='yuv420p'
        except BaseException:
            self.container.close()
            raise

    def write(self,frame):
        image=self.av.VideoFrame.from_ndarray(frame,format='bgr24')
        for packet in self.stream.encode(image):self.container.mux(packet)

    def release(self):
        try:
            for packet in self.stream.encode(None):self.container.mux(packet)
        finally:
            self.container.close()

def load_display_labels(path):
    """Load independent sequence labels for display only, never inference."""
    source=json.loads(path.read_text())
    if not isinstance(source,dict) or not isinstance(source.get('frames'),list):
        raise ValueError('Display labels must be a sequence JSON with a frames list')
    labels={}
    for frame in source['frames']:
        index=frame.get('frame_index')
        if type(index) is not int or index<0 or index in labels:
            raise ValueError('Display label frame indices must be unique nonnegative integers')
        caps=sorted({d['slot_id'] for d in frame.get('annotations',[])
                     if d.get('category_id') in (2,3) and d.get('slot_id','').startswith('G')})
        gaps=sorted({sid for sid in frame.get('gap_slots',[]) if isinstance(sid,str) and sid.startswith('G')})
        labels[index]={'cap_defects':caps,'missing_vials':gaps}
    return source,labels


def draw_defect_summary(frame,status,actual):
    """Always display model decisions; label actuals unavailable when absent."""
    def slots(values):return ', '.join(sorted(set(values))) or 'none'
    def description(caps,missing):return 'caps ['+slots(caps)+'] | missing ['+slots(missing)+']'
    actual_text='unavailable' if actual is None else description(actual['cap_defects'],actual['missing_vials'])
    raw=[d['slot_id'] for d in status['raw_cap_defects']]
    confirmed=[d['slot_id'] for d in status['cap_defects']]
    entries=[('Actual: '+actual_text,(255,200,0)),
             ('Model: '+description(raw,status['missing_slots']),(255,255,255)),
             ('Confirmed: '+description(confirmed,status['missing_slots']),(180,255,180))]
    # Wrap by rendered width so a long defect list stays inside the frame.
    height,width=frame.shape[:2];lines=[];font=cv2.FONT_HERSHEY_SIMPLEX;scale=.4
    for text,color in entries:
        line=''
        for word in text.split():
            candidate=(line+' '+word).strip()
            if line and cv2.getTextSize(candidate,font,scale,1)[0][0]>width-24:
                lines.append((line,color));line='  '+word
            else:line=candidate
        lines.append((line,color))
    panel_height=17*len(lines)+12;top=max(0,height-panel_height)
    shade=frame.copy();cv2.rectangle(shade,(0,top),(width-1,height-1),(0,0,0),-1)
    cv2.addWeighted(shade,.75,frame,.25,0,frame)
    for index,(text,color) in enumerate(lines):
        cv2.putText(frame,text,(10,top+17*(index+1)),font,scale,color,1,cv2.LINE_AA)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model',type=Path,help='TorchScript detector: float RGB BCHW in [0,1], output Nx6 xyxy/score/category in input pixels')
    mode.add_argument('--predictions',type=Path,help='External detector JSON records: frame_index, bbox xywh, category_id, score')
    mode.add_argument('--ground-truth',type=Path,help='Ground-truth JSON: an oracle prediction list or a sequence JSON with per-frame annotations')
    p.add_argument('--labels',type=Path,help='Independent sequence labels for the Actual summary; never used as model input. Auto-detects ../sequences/<video>.json when available.')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--confirm',type=int,default=3)
    p.add_argument('--threshold',type=float,default=.1,help='Cap-defect confidence cutoff')
    p.add_argument('--presence-threshold',type=float,default=.25,help='Vial presence cutoff, separate from cap defects')
    p.add_argument('--absence-frames',type=int,default=3,help='Consecutive absent observations before declaring a slot empty')
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.output.exists():p.error('Output exists')
    labels_path=args.labels
    if labels_path is None:
        candidate=args.video.resolve().parent.parent/'sequences'/(args.video.stem+'.json')
        if candidate.is_file():labels_path=candidate
        elif args.ground_truth and args.ground_truth.is_file():
            candidate_source=json.loads(args.ground_truth.read_text())
            if isinstance(candidate_source,dict) and isinstance(candidate_source.get('frames'),list):labels_path=args.ground_truth
    label_source,display_labels=load_display_labels(labels_path) if labels_path is not None else ({},{})
    calibration=json.loads(args.calibration.read_text());slots=calibration['slots']
    tracker=VideoInspector(slots,args.confirm,args.absence_frames,args.threshold,args.presence_threshold)
    # Calibrated fallback boxes locate slots even when no vial is detected.
    slot_boxes={}
    for slot in tracker.slots:
        center=slot.get('inspection_anchor_native',slot['center_native'])
        distances=[np.linalg.norm(np.asarray(center)-np.asarray(other.get('inspection_anchor_native',other['center_native'])))
                   for other in tracker.slots if other['slot_id']!=slot['slot_id']]
        side=float(min(distances))*.8
        slot_boxes[slot['slot_id']]=[center[0]-side/2,center[1]-side/2,side,side]
    predictions=defaultdict(list)
    if args.predictions or args.ground_truth:
        source=json.loads((args.predictions or args.ground_truth).read_text())
        if args.ground_truth and isinstance(source,dict):
            if not isinstance(source.get('frames'),list):
                raise ValueError('Ground-truth JSON must contain a frames list')
            source=[dict(annotation,frame_index=frame['frame_index'])
                    for frame in source['frames']
                    for annotation in frame.get('annotations',[])]
        if not isinstance(source,list):
            raise ValueError('Prediction and ground-truth JSON must contain a list of records')
        for d in source:
            if not isinstance(d['frame_index'],int) or d['frame_index']<0:raise ValueError('Prediction frame_index must be nonnegative integer')
            if args.ground_truth and 'score' not in d:d['score']=1.0
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
    if label_source.get('fps') is not None and abs(float(label_source['fps'])-fps)>.001:
        capture.release();raise ValueError('Display label FPS differs from video FPS')
    expected_frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if label_source.get('frame_count') is not None and expected_frames>0 and label_source['frame_count']!=expected_frames:
        capture.release();raise ValueError('Display label frame count differs from video')
    args.output.mkdir(parents=True);writer=None;count=0;complete=0;missing_frames=0
    try:
        with (args.output/'frame-results.jsonl').open('w') as results:
            while True:
                ok,frame=capture.read()
                if not ok:break
                height,width=frame.shape[:2]
                if [width,height]!=calibration['image_size']:raise ValueError('Video and calibration coordinates differ')
                if args.predictions or args.ground_truth:detections=predictions.get(count,[])
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
                status=tracker.update(assigned,count,count/fps)
                status['raw_cap_defects']=[d for d in assigned if d['category_id'] in (2,3) and d['score']>=args.threshold]
                actual=display_labels.get(count)
                status['actual_defects']=actual
                status['ground_truth_available']=actual is not None
                results.write(json.dumps(status)+'\n')
                complete+=status['order_status']=='complete'
                missing_frames+=bool(status['missing_slots'])
                cv2.putText(frame,f"G: {status['occupied_count']}/30 {status['order_status']}",(12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0) if status['order_status']=='complete' else (0,180,255),2)
                # One box per detected slot. A defect wins over presence even
                # when the presence score is higher; category 1 alone means no
                # defect detected above the cutoff, not independently verified good.
                overlays={}
                for d in assigned:
                    cutoff=args.presence_threshold if d['category_id']==1 else args.threshold
                    if d['category_id'] not in (1,2,3) or d['score']<cutoff:continue
                    previous=overlays.get(d['slot_id'])
                    priority=(d['category_id']!=1,d['score'])
                    if previous is None or priority>(previous['category_id']!=1,previous['score']):
                        overlays[d['slot_id']]=d
                for slot in tracker.slots:
                    sid=slot['slot_id'];d=overlays.get(sid)
                    if sid in status['missing_slots']:
                        bbox=slot_boxes[sid];color=(0,0,255);label=f'{sid} MISSING'
                    elif d is not None and d['category_id']!=1:
                        bbox=d['bbox'];color=(0,0,255);label=f"{sid} BAD {d['score']:.2f}"
                    elif sid in status['unknown_slots']:
                        bbox=slot_boxes[sid];color=(0,220,255);label=f'{sid} UNKNOWN'
                    elif d is not None and sid in status['occupied_slots']:
                        bbox=d['bbox'];color=(0,255,0);label=f"{sid} GOOD {d['score']:.2f}"
                    else:continue  # Normal trailing empties have no defect box.
                    x,y,w,h=map(round,bbox)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                    cv2.putText(frame,label,(x,max(12,y-3)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
                draw_defect_summary(frame,status,actual)
                if writer is None:
                    writer=H264VideoWriter(args.output/'inspection.mp4',fps,(width,height))
                writer.write(frame);count+=1
    finally:
        capture.release()
        if writer:writer.release()
    if not count:raise ValueError('Video contained no decoded frames')
    if (args.predictions or args.ground_truth) and any(index>=count for index in predictions):
        raise ValueError('Predictions reference frames outside the video')
    (args.output/'summary.json').write_text(json.dumps({'frames':count,'fps':fps,'complete_frames':complete,
        'missing_vial_frames':missing_frames,'presence_threshold':args.presence_threshold,'absence_frames':args.absence_frames,
        'display_labels':str(labels_path.resolve()) if labels_path is not None else None,
        'defect_summary':'Actual = independent labels (or unavailable); Model = raw cap predictions plus confirmed missing-vial alerts; Confirmed = temporal cap and missing-vial alerts.',
        'output_codec':'h264','output_pixel_format':'yuv420p',
        'mode':'ground_truth_replay' if args.ground_truth else ('external_prediction_replay' if args.predictions else 'trained_detector'),
        'model':str(args.model) if args.model else None,'threshold':args.threshold,'confirmation_frames':args.confirm,
        'overlay':'Red MISSING: confirmed fill-order gap or disappearance of a previously occupied vial. Red BAD: raw cap-defect candidate. Yellow UNKNOWN: unconfirmed presence/absence. Green GOOD: confirmed presence without an above-threshold cap defect; not verified defect-free. Normal trailing empties have no box.',
        'note':'Ground-truth replay; this validates annotation/video plumbing, not detector accuracy.' if args.ground_truth
            else ('External prediction replay; no accuracy score is computed.' if args.predictions
                  else 'Learned detector used; complete-frame counts are outputs, not ground-truth accuracy scores.')},indent=2)+'\n')
if __name__=='__main__':main()
