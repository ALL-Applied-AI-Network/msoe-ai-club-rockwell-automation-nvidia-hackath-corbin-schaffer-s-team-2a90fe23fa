#!/usr/bin/env python3
"""Export full frames and scene-split COCO annotations from verified placement sidecars."""
import argparse,json,shutil
from collections import Counter,defaultdict
from statistics import median
from pathlib import Path
from PIL import Image,ImageDraw
from full_frame_detection import frame_annotations,CATEGORIES

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--placements',type=Path,required=True)
    parser.add_argument('--splits',type=Path,required=True,help='Existing scene assignments; no random resplitting')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scope',choices=['G','both'],default='G')
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; choose a new directory')
    splits=json.loads(args.splits.read_text());sources=sorted(args.placements.glob('*.placements.json'))
    if not sources:parser.error('No placement sidecars')
    # Validate all source paths/assignments before creating the export.
    for p in sources:
        if splits.get(p.name) not in ('train','validation','test'):parser.error(f'Missing split for {p.name}')
        data=json.loads(p.read_text());frame_annotations(data,args.scope)
        if not (p.parent/data['image']).is_file():parser.error(f'Missing full image for {p.name}')
    for directory in ['images','annotations','slots','qa']:(args.output/directory).mkdir(parents=True,exist_ok=True)
    sets={s:{'images':[],'annotations':[],'categories':CATEGORIES} for s in ['train','validation','test']}
    counts=Counter();annotation_id=1;next_overlay=Counter()
    calibration={};anchors=defaultdict(list)
    for image_id,p in enumerate(sources,1):
        meta=json.loads(p.read_text());frame=frame_annotations(meta,args.scope);split=splits[p.name]
        path=p.parent/meta['image']
        with Image.open(path) as im:
            assert list(im.size)==meta['native_size'],f'{path}: expected native full frame'
            overlay=im.convert('RGB')
        shutil.copy2(path,args.output/'images'/path.name)
        sets[split]['images'].append({'id':image_id,'file_name':path.name,'width':overlay.width,'height':overlay.height,
                                    'slot_metadata':f'slots/{p.stem}.json','inspection_scope':args.scope,
                                    'good_tray_order':frame['good_tray_order']})
        for slot in frame['slots']:
            calibration.setdefault(slot['slot_id'],{'slot_id':slot['slot_id'],'tray':slot['tray'],'position':slot['position'],'center_native':slot['center_native']})
            if slot['present'] and slot['label']=='PASS':
                x0,y0,x1,y1=slot['inspection_bounds_native'];anchors[slot['slot_id']].append(((x0+x1)/2,(y0+y1)/2))
        draw=ImageDraw.Draw(overlay)
        for ann in frame['annotations']:
            ann=dict(ann,id=annotation_id,image_id=image_id);annotation_id+=1
            sets[split]['annotations'].append(ann);counts[f'{split}/{CATEGORIES[ann["category_id"]-1]["name"]}']+=1
            if ann['category_id']!=1:
                x,y,w,h=ann['bbox'];draw.rectangle((x,y,x+w,y+h),outline='red',width=2)
                draw.text((x,y),ann['slot_id']+' '+CATEGORIES[ann['category_id']-1]['name'],fill='red')
        for slot in frame['slots']:
            x,y=slot['center_native'];draw.text((x,y),slot['slot_id'],fill='yellow' if slot['gap_error'] else 'lime')
        (args.output/'slots'/f'{p.stem}.json').write_text(json.dumps(frame,indent=2)+'\n')
        if next_overlay[split]<3:
            overlay.save(args.output/'qa'/f'{split}_{path.name}');next_overlay[split]+=1
        counts[split+'/images']+=1;counts[split+'/gap_errors']+=len(frame['gap_errors'])
        counts[split+'/order_'+frame['good_tray_order']['order_status']]+=1
    for split,data in sets.items():
        (args.output/'annotations'/f'{split}.json').write_text(json.dumps(data,indent=2)+'\n')
    for name,slot in calibration.items():
        points=anchors[name]
        if not points:raise ValueError(f'No normal-vial anchor for {name}')
        slot['inspection_anchor_native']=[median(p[0] for p in points),median(p[1] for p in points)]
        slot['normal_observations']=len(points)
    (args.output/'calibration.json').write_text(json.dumps({'image_size':[640,480],'scope':args.scope,'slots':list(calibration.values()),'note':'Fixed synthetic camera only; recalibrate for a real camera or changed view.'},indent=2)+'\n')
    summary={'status':'exported_pending_detection_QA','scope':args.scope,'counts':dict(counts),
             'coordinate_system':'native 640x480 full frame; no crops/resizing',
             'slot_ids':'G01-G30; S01-S30 only with --scope both',
             'limitations':['Boxes are projected amodal mouth/cap bounds, not visible defect segmentation.',
                            'Slot attributes are metadata; a standard detector does not learn arbitrary slot IDs.',
                            'G-only training intentionally treats the S tray as out of inspection scope.',
                            'Training a detector has not been performed.']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
