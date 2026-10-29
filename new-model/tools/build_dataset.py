"""Convert placement-tagged native CapSeal renders into a scene-split TAO dataset."""
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps


def gap_positions(occupied):
    """Positions are 1-based, top-to-bottom then left, within one tray."""
    return sorted(set(range(1, max(occupied, default=0))) - set(occupied))


def build(renders, golden, output):
    sidecars = sorted(renders.glob('*.placements.json'))
    if len(sidecars) < 3:
        raise ValueError('Need at least three scenes to populate all splits')
    if output.exists():
        raise ValueError(f'Refusing to overwrite {output}')
    golden_bytes = golden.read_bytes()
    golden_hash = hashlib.sha256(golden_bytes).hexdigest()
    output.mkdir(parents=True)
    for sub in ('images/inspection','images/golden','csv','qa'):
        (output/sub).mkdir(parents=True)
    # Deterministic scene ordering; every crop from a scene stays in its scene split.
    scenes = sorted(sidecars, key=lambda p: hashlib.sha256(p.stem.encode()).hexdigest())
    ntest = max(1, round(len(scenes)*.15))
    nval = max(1, round(len(scenes)*.15))
    assignments = {p: ('test' if i < ntest else 'validation' if i < ntest+nval else 'train')
                   for i,p in enumerate(scenes)}
    rows = {k: [] for k in ('train','validation','test')}
    manifest, counts, sources, fingerprints = [], Counter(), {}, {}
    thumbnails = []
    for sidecar in sidecars:
        info = json.loads(sidecar.read_text())
        image_path = sidecar.parent/info['image']
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if image_hash in fingerprints:
            raise ValueError(f'Duplicate render: {image_path} and {fingerprints[image_hash]}')
        fingerprints[image_hash] = image_path.name
        with Image.open(image_path) as original:
            if original.size != (640,480):
                raise ValueError(f'Expected native 640x480 image: {image_path}')
            image = original.convert('RGB').resize((1920,1440),Image.Resampling.LANCZOS).crop((0,360,1920,1440))
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for tray in ('G','S'):
            slots = [s for s in info['slots'] if s['tray']==tray]
            expected = gap_positions([s['position'] for s in slots if s['present']])
            assert expected == [s['position'] for s in slots if s['gap_error']]
        for slot in info['slots']:
            bounds = slot.get('inspection_bounds_native')
            if slot['present'] and not bounds:
                raise ValueError('Missing inspection geometry; regenerate placement sidecars with the updated renderer')
            if bounds:
                x0,y0,x1,y1 = bounds
                cx,cy = (x0+x1)/2,(y0+y1)/2
                # Contain the projected mouth and displaced/rotated cap with 15% padding.
                width = max(30, (x1-x0)*1.3, (y1-y0)*1.3*90/108)
                height = width*108/90
            else:
                cx,cy = slot['center_native']
                width,height = 30,36
            cx,cy = (cx+.5)*3-.5, (cy+.5)*3-.5-360
            box = [round(cx-width*1.5),round(cy-height*1.5),
                   round(cx+width*1.5),round(cy+height*1.5)]
            record = dict(slot, scene=image_path.name, split=assignments[sidecar], bbox=box,
                          seed=info['seed'], variation=info['variation'])
            record['excluded'] = not slot['present'] or box[0]<0 or box[1]<0 or box[2]>1920 or box[3]>1080
            manifest.append(record)
            color = 'yellow' if slot['gap_error'] else 'lime' if slot['label']=='PASS' else 'red'
            draw.rectangle(box,outline=color,width=2)
            draw.text((box[0],box[1]),slot['slot_id'],fill=color)
            counts['gaps'] += int(slot['gap_error'])
            if record['excluded']:
                counts['excluded'] += 1
                continue
            sample = f'{image_path.stem}_{slot["slot_id"]}'
            filename = sample+'_SolderLight.jpg'
            crop = ImageOps.pad(image.crop(box),(90,108),method=Image.Resampling.LANCZOS)
            crop.save(output/'images/inspection'/filename,quality=95)
            (output/'images/golden'/filename).write_bytes(golden_bytes)
            record['sample_id'] = sample
            rows[record['split']].append(['inspection','golden',slot['label'],sample])
            counts[record['split']+'/'+slot['label']] += 1
            sources.setdefault(slot['source_id'],set()).add(record['split'])
            if len(thumbnails)<60:
                thumb = ImageOps.pad(crop,(90,108))
                ImageDraw.Draw(thumb).text((1,1),slot['label'],fill='yellow')
                thumbnails.append(thumb)
        if len(sidecars)<=6 or sidecar in scenes[:3]:
            overlay.save(output/'qa'/f'{image_path.stem}_overlay.jpg')
    for split, values in rows.items():
        with (output/'csv'/f'{split}.csv').open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(['input_path','golden_path','label','object_name']);writer.writerows(values)
    shutil.copyfile(output/'csv/test.csv',output/'csv/inference.csv')
    with (output/'manifest.jsonl').open('w') as f:
        for record in manifest: f.write(json.dumps(record)+'\n')
    sheet=Image.new('RGB',(900,108*((len(thumbnails)+9)//10)))
    for i,thumb in enumerate(thumbnails): sheet.paste(thumb,((i%10)*90,(i//10)*108))
    sheet.save(output/'qa/contact_sheet.jpg')
    metadata={'scenes':len(sidecars),'golden':str(golden.resolve()),'golden_sha256':golden_hash,
              'counts':dict(counts),'status':'pilot_requires_visual_review',
              'crop_policy':'projected mouth and cap bounds with 15% padding, aspect preserved',
              'crop_size':[90,108], 'golden_native_size':list(Image.open(golden).size),
              'source_overlap_across_splits':{k:sorted(v) for k,v in sources.items() if len(v)>1},
              'limitations':['Labels use renderer source-name/tilt heuristic; visual QA required.',
                            'Scene-disjoint splits share source assembly families; not independent defect families.',
                            'Fixed real golden and synthetic crops need framing/scale review.'],
              'occupancy_policy':'empty before last occupied position is gap; trailing empties pending; per tray'}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'splits.json').write_text(json.dumps({p.name:s for p,s in assignments.items()},indent=2)+'\n')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renders',type=Path,required=True)
    parser.add_argument('--golden',type=Path,default=Path('new-model/GoldenImages/G24.jpg'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.renders,args.golden,args.output)
