#!/usr/bin/env python3
"""Render nine independent scene sequences, encode videos and export frame/COCO/event labels."""
import argparse,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
from collections import Counter
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'csi_digital_twin_v1/tools'))
from full_frame_detection import frame_annotations,CATEGORIES

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'final_model/dataset-v2')
    p.add_argument('--fps',type=int,default=5);p.add_argument('--hold-frames',type=int,default=4)
    p.add_argument('--train-defect-probability',type=float,default=.5)
    p.add_argument('--eval-defect-probability',type=float,default=.1)
    p.add_argument('--train-sequences',type=int,default=3)
    p.add_argument('--validation-sequences',type=int,default=3)
    p.add_argument('--test-sequences',type=int,default=3)
    p.add_argument('--seed-base',type=int,default=92000)
    p.add_argument('--vary-sequences',action='store_true')
    args=p.parse_args();output=args.output.resolve()
    split_counts={'train':args.train_sequences,'validation':args.validation_sequences,'test':args.test_sequences}
    if min(split_counts.values())<3:p.error('At least three sequences per split are required')
    total_sequences=sum(split_counts.values())
    if not 0<=args.eval_defect_probability<args.train_defect_probability<=1:
        p.error('Require 0 <= eval probability < train probability <= 1')
    if output.exists():p.error('Output already exists')
    if args.fps<1 or args.hold_frames<3:p.error('FPS must be positive; hold at least 3 frames for temporal confirmation')
    output.mkdir(parents=True)
    for d in ['videos','frames','annotations','sequences','render-logs','raw-states']:(output/d).mkdir()
    (output/'generation-config.json').write_text(json.dumps({'split_sequences':split_counts,'train_defect_probability':args.train_defect_probability,'eval_defect_probability':args.eval_defect_probability,'seed_base':args.seed_base,'vary_sequences':args.vary_sequences,'fps':args.fps,'hold_frames':args.hold_frames},indent=2)+'\n')
    status=output/'status.json'
    def state(stage,**kw):
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(stage=stage,**kw),indent=2)+'\n');tmp.replace(status)
    coco={s:{'images':[],'annotations':[],'categories':CATEGORIES} for s in ['train','validation','test']}
    sequence_index=[];image_id=0;ann_id=0;defect_rates={}
    shutil.copy2(ROOT/'new-model/dataset-detection-G/calibration.json',output/'calibration.json')
    try:
        for split_idx,split in enumerate(coco):
            for scenario_idx in range(split_counts[split]):
                scenario=['fill','gap','fill_pause'][scenario_idx%3]
                sequence_id=f'{split}_{scenario}_{scenario_idx:03d}' if split_counts[split]>3 else f'{split}_{scenario}'
                seed=args.seed_base+split_idx*1000000+scenario_idx
                probability=args.train_defect_probability if split=='train' else args.eval_defect_probability
                state('RENDERING',sequence=sequence_id,completed_sequences=len(sequence_index),target_sequences=total_sequences)
                raw=output/'raw-states'/sequence_id;raw.mkdir()
                command=[sys.executable,str(ROOT/'csi_digital_twin_v1/tools/isaac_headless.py'),str(ROOT/'final_model/tools/render_video_states.py'),
                         '--camera','CapSealCam','--stage','stations/tray_inspection/tray_inspection.usdc','--randomize-trays',
                         '--variations','8','--min-vials','0','--max-vials','30','--seed',str(seed),
                         '--defect-probability',str(probability),'--export-detections','--subframes','16','--sequence-scenario',scenario,'--out',str(raw/'state.png')]
                if args.vary_sequences:command.append('--vary-sequence')
                with (output/'render-logs'/f'{sequence_id}.log').open('w') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                sidecars=sorted(raw.glob('*.placements.json'))
                if len(sidecars)!=8:raise ValueError('Expected 8 rendered scene states')
                video=cv2.VideoWriter(str(output/'videos'/f'{sequence_id}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(640,480))
                if not video.isOpened():raise RuntimeError('MP4 encoder unavailable')
                frames_dir=output/'frames'/sequence_id;frames_dir.mkdir()
                frames=[];events=[];oracle_predictions=[];frame_index=0;last_order=None
                try:
                    for state_idx,path in enumerate(sidecars):
                        meta=json.loads(path.read_text());full=frame_annotations(meta,'G')
                        if state_idx!=meta['sequence_state']:raise ValueError('State order mismatch')
                        image=cv2.imread(str(raw/meta['image']))
                        if image is None or image.shape[:2]!=(480,640):raise ValueError('Bad rendered frame')
                        events.append({'state_index':state_idx,'frame_index':frame_index,'timestamp_seconds':frame_index/args.fps,
                                       'event':'placement_state','occupied_count':full['good_tray_order']['occupied_count'],
                                       'order_status':full['good_tray_order']['order_status'],'gap_slots':full['gap_errors'],
                                       'defects':[{'slot_id':a['slot_id'],'category_id':a['category_id']} for a in full['annotations'] if a['category_id']!=1]})
                        for repeat in range(args.hold_frames):
                            name=f'{frame_index:06d}.png';relative=f'{sequence_id}/{name}';shutil.copy2(raw/meta['image'],frames_dir/name)
                            video.write(image);image_id+=1
                            coco[split]['images'].append({'id':image_id,'file_name':relative,'width':640,'height':480,
                                'sequence_id':sequence_id,'frame_index':frame_index,'timestamp_seconds':frame_index/args.fps,
                                'good_tray_order':full['good_tray_order']})
                            objects=[]
                            for ann in full['annotations']:
                                ann_id+=1;annot=dict(ann,id=ann_id,image_id=image_id);coco[split]['annotations'].append(annot)
                                objects.append(ann)
                                oracle_predictions.append({'frame_index':frame_index,'bbox':ann['bbox'],'category_id':ann['category_id'],'score':1.0})
                            frames.append({'frame_index':frame_index,'timestamp_seconds':frame_index/args.fps,'state_index':state_idx,
                                'file_name':relative,'good_tray_order':full['good_tray_order'],'gap_slots':full['gap_errors'],
                                'slots':full['slots'],'annotations':objects})
                            frame_index+=1
                finally:video.release()
                seq={'sequence_id':sequence_id,'split':split,'seed':seed,'scenario':scenario,'fps':args.fps,'frame_count':frame_index,
                     'video':f'videos/{sequence_id}.mp4','events':events,'frames':frames,
                     'defect_probability':probability,'defect_probability_unit':'unique placed vial; sampled once and held through sequence',
                     'defect_plan':meta['defect_plan'],'skipped_slot':meta.get('skipped_slot'),
                     'fill_counts':meta.get('fill_counts'),'light_intensity_factor':meta.get('light_intensity_factor',1),
                     'limitations':['Discrete static placement states held across frames; no robot/flight dynamics or physically integrated motion blur.',
                                    'Fixed synthetic camera; shared geometry source families across sequences.']}
                (output/'sequences'/f'{sequence_id}.json').write_text(json.dumps(seq,indent=2)+'\n')
                (output/'sequences'/f'{sequence_id}.oracle-predictions.json').write_text(json.dumps(oracle_predictions)+'\n')
                final=frames[-1]
                actual=sum(s['present'] and s['label']=='NO_PASS' for s in final['slots'])
                present=sum(s['present'] for s in final['slots'])
                defect_rates[sequence_id]={'split':split,'requested_probability':probability,'placed_vials':present,'defective_vials':actual,'observed_unique_vial_rate':actual/present}
                (output/'defect-rate-summary.json').write_text(json.dumps(defect_rates,indent=2)+'\n')
                sequence_index.append({k:seq[k] for k in ['sequence_id','split','seed','scenario','fps','frame_count','video']})
                (output/'sequence-index.json').write_text(json.dumps(sequence_index,indent=2)+'\n')
                for split_name,data in coco.items():(output/'annotations'/f'{split_name}.json').write_text(json.dumps(data,indent=2)+'\n')
        # Detector training consumes unique placement states, not held-frame copies.
        for split_name,data in coco.items():
            keep=set()
            for sequence in [s for s in sequence_index if s['split']==split_name]:
                details=json.loads((output/'sequences'/f"{sequence['sequence_id']}.json").read_text());seen=set()
                for frame in details['frames']:
                    if frame['frame_index']%args.hold_frames:continue
                    signature=json.dumps([(s['slot_id'],s['present'],s.get('cap_present'),s.get('cap_tilt_degrees')) for s in frame['slots']])
                    if signature in seen:continue
                    seen.add(signature);keep.add((sequence['sequence_id'],frame['frame_index']))
            images=[im for im in data['images'] if (im['sequence_id'],im['frame_index']) in keep];ids={im['id'] for im in images}
            (output/'annotations'/f'{split_name}-keyframes.json').write_text(json.dumps({'images':images,'annotations':[a for a in data['annotations'] if a['image_id'] in ids],'categories':CATEGORIES},indent=2)+'\n')
        state('COMPLETE_NOT_TRAINED',sequences=total_sequences,frames=sum(s['frame_count'] for s in sequence_index),rendered_states=8*total_sequences,
              note='Initial coherent synthetic video dataset; repeated held frames are not independent observations.')
    except BaseException:
        state('ERROR',traceback=traceback.format_exc());raise
if __name__=='__main__':main()
