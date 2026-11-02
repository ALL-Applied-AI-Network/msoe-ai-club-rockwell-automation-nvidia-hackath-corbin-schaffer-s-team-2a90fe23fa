#!/usr/bin/env python3
"""Attach G-slot IDs to native-frame COCO detector predictions; infer fill-order gaps."""
import argparse,json
from collections import defaultdict
from pathlib import Path
from full_frame_detection import assign_to_slot,CATEGORIES,good_tray_order_status

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True,help='COCO results: image_id, bbox xywh, category_id, score')
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--images',type=Path,help='COCO dataset image manifest; includes frames with zero predictions')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--score-threshold',type=float,default=0.1,help='Candidate cutoff; must be calibrated on validation data')
    p.add_argument('--scope',choices=['G','both'],default='G')
    args=p.parse_args()
    if args.output.exists():p.error('Refusing to overwrite output')
    if not 0<=args.score_threshold<=1:p.error('Threshold must be in [0,1]')
    calibration=json.loads(args.calibration.read_text());slots=calibration['slots']
    groups=defaultdict(list);unassigned=[];ids=set()
    if args.images:
        ids={image['id'] for image in json.loads(args.images.read_text())['images']}
    manifest_ids=set(ids)
    for item in json.loads(args.predictions.read_text()):
        if args.images and item['image_id'] not in manifest_ids:raise ValueError('Prediction image is absent from image manifest')
        ids.add(item['image_id'])
        if item['score']<args.score_threshold:continue
        if item['category_id'] not in (1,2,3):raise ValueError('Unknown category id')
        slot=assign_to_slot(item['bbox'],slots,args.scope)
        if slot is None:unassigned.append(item);continue
        groups[item['image_id']].append(dict(item,slot_id=slot,defect=CATEGORIES[item['category_id']-1]['name']))
    frames=[]
    for image_id in sorted(ids):
        annotations=groups[image_id]
        present={a['slot_id'] for a in annotations} # defects also imply a present vial
        gaps=[]
        for tray in ('G','S') if args.scope=='both' else ('G',):
            tray_slots=[s for s in slots if s['tray']==tray]
            last=max([s['position'] for s in tray_slots if s['slot_id'] in present],default=0)
            gaps.extend(s['slot_id'] for s in tray_slots if s['position']<last and s['slot_id'] not in present)
        frames.append(dict(image_id=image_id,detections=annotations,defects=[a for a in annotations if a['category_id']!=1],possible_gap_slots=gaps,good_tray_order=good_tray_order_status(present)))
    args.output.write_text(json.dumps({'frames':frames,'unassigned':unassigned,'threshold':args.score_threshold,
        'occupancy_basis':'confidence-filtered detector predictions; undetected G slots are reported empty/incomplete',
        'note':'Boxes must be restored to calibration native pixel coordinates after detector resizing. Gaps are candidate occupancy errors, sensitive to missed vial detections; not temporal machine-state proof.'},indent=2)+'\n')
if __name__=='__main__':main()
