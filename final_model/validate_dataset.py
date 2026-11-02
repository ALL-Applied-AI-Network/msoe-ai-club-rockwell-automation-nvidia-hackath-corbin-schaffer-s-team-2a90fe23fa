#!/usr/bin/env python3
"""Check video decoding, temporal ground truth and sequence-disjoint COCO exports."""
import argparse,json,hashlib,sys,subprocess
from pathlib import Path
from collections import Counter
import cv2
from PIL import Image,ImageDraw

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=Path('final_model/dataset-v2'));args=p.parse_args();root=args.dataset.resolve()
    status=json.loads((root/'status.json').read_text())
    if status['stage']!='COMPLETE_NOT_TRAINED':raise ValueError('Generation has not completed')
    sequences=json.loads((root/'sequence-index.json').read_text());assert len(sequences)>=9
    assert len({s['seed'] for s in sequences})==len(sequences)
    image_ids=set();annotation_ids=set();sequence_splits={};counts=Counter();report=[]
    for split in ('train','validation','test'):
        coco=json.loads((root/'annotations'/f'{split}.json').read_text());ids=set()
        for image in coco['images']:
            assert image['id'] not in image_ids;ids.add(image['id']);image_ids.add(image['id'])
            seq=image['sequence_id'];assert sequence_splits.get(seq,split)==split;sequence_splits[seq]=split
            with Image.open(root/'frames'/image['file_name']) as im:assert im.size==(640,480)
        for ann in coco['annotations']:
            assert ann['id'] not in annotation_ids;annotation_ids.add(ann['id']);assert ann['image_id'] in ids
            assert ann['slot_id'].startswith('G');x,y,w,h=ann['bbox'];assert 0<=x<x+w<=640 and 0<=y<y+h<=480
    for seq in sequences:
        data=json.loads((root/'sequences'/f"{seq['sequence_id']}.json").read_text());identities={};last=0
        for frame in data['frames']:
            assert frame['timestamp_seconds']==frame['frame_index']/data['fps']
            present=[s for s in frame['slots'] if s['present']];occupied={s['slot_id'] for s in present}
            assert len(present)>=last;last=len(present)
            for slot in present:
                assert identities.get(slot['slot_id'],slot['source_id'])==slot['source_id'];identities[slot['slot_id']]=slot['source_id']
            assert frame['good_tray_order']['occupied_count']==len(present)
            assert frame['good_tray_order']['order_complete']==(len(occupied)==30)
            boundary=max([s['position'] for s in present],default=0)
            gaps={s['slot_id'] for s in frame['slots'] if not s['present'] and s['position']<boundary}
            assert gaps==set(frame['gap_slots'])
            counts['complete_ground_truth_frames' if len(occupied)==30 else 'incomplete_ground_truth_frames']+=1
        final=data['frames'][-1]
        if seq['scenario']=='gap':assert final['gap_slots']==[data.get('skipped_slot') or 'G12'] and not final['good_tray_order']['order_complete']
        else:assert final['good_tray_order']['order_complete']
        if 'defect_probability' not in data and seq['scenario']=='normal':assert not any(a['category_id']!=1 for f in data['frames'] for a in f['annotations'])
        if 'defect_probability' not in data and seq['scenario']=='cap_defects':assert {(a['slot_id'],a['category_id']) for a in final['annotations'] if a['category_id']!=1}=={('G08',2),('G18',3)}
        if 'defect_probability' in data:
            expected={(name,2 if defect['kind']=='tilt' else 3) for name,defect in data['defect_plan'].items() if name in occupied}
            assert {(a['slot_id'],a['category_id']) for a in final['annotations'] if a['category_id']!=1}==expected
            for frame in data['frames']:
                seen={s['slot_id'] for s in frame['slots'] if s['present']}
                assert {(a['slot_id'],a['category_id']) for a in frame['annotations'] if a['category_id']!=1}=={pair for pair in expected if pair[0] in seen}
        cap=cv2.VideoCapture(str(root/seq['video']));decoded=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape[:2]==(480,640);decoded+=1
        cap.release();assert decoded==seq['frame_count']==len(data['frames'])
        # Oracle replay checks video reading, mapping and temporal state, not model accuracy.
        replay=root/'qa'/seq['sequence_id'];replay.parent.mkdir(exist_ok=True)
        subprocess.run([sys.executable,str(Path(__file__).with_name('inspect_video.py')),'--video',str(root/seq['video']),
                        '--calibration',str(root/'calibration.json'),'--predictions',str(root/'sequences'/f"{seq['sequence_id']}.oracle-predictions.json"),
                        '--output',str(replay)],check=True)
        predicted=[json.loads(line) for line in (replay/'frame-results.jsonl').read_text().splitlines()]
        last_prediction=predicted[-1]
        assert last_prediction['order_status']==final['good_tray_order']['order_status'],seq
        assert set(last_prediction['gap_slots'])==set(final['gap_slots']),seq
        if 'defect_probability' not in data and seq['scenario']=='cap_defects':assert {(a['slot_id'],a['category_id']) for a in last_prediction['defects']}=={('G08',2),('G18',3)}
        if 'defect_probability' in data:
            assert {(a['slot_id'],a['category_id']) for a in last_prediction['defects']}==expected
        report.append({'sequence':seq['sequence_id'],'decoded_frames':decoded,'final_order':last_prediction['order_status'],'gap_slots':last_prediction['gap_slots']})
    sheet=Image.new('RGB',(960,240*min(len(sequences),18)),'white');draw=ImageDraw.Draw(sheet)
    for row,seq in enumerate(sequences[:18]):
        data=json.loads((root/'sequences'/f"{seq['sequence_id']}.json").read_text())
        for col,index in enumerate([0,12,28]):
            frame=data['frames'][index]
            with Image.open(root/'frames'/frame['file_name']) as im:sheet.paste(im.resize((320,240)),(col*320,row*240))
            draw.text((col*320+4,row*240+4),f"{seq['sequence_id']} f{index} {frame['good_tray_order']['order_status']}",fill='red')
    sheet.save(root/'qa/contact-sheet.jpg')
    summary={'status':'validated_no_training','sequences':len(sequences),'frames':len(image_ids),'rendered_states':8*len(sequences),'counts':dict(counts),
             'checks':report,'oracle_replay_note':'Ground-truth detections only validate video/slot/temporal plumbing; no learned detector accuracy measured.'}
    if (root/'defect-rate-summary.json').exists():
        rates=json.loads((root/'defect-rate-summary.json').read_text());summary['defect_rates']={}
        for split in ('train','validation','test'):
            rows=[r for r in rates.values() if r['split']==split];n=sum(r['placed_vials'] for r in rows);bad=sum(r['defective_vials'] for r in rows)
            summary['defect_rates'][split]={'placed_vials':n,'defective_vials':bad,'observed_rate':bad/n,'requested_probability':rows[0]['requested_probability']}
        assert summary['defect_rates']['train']['observed_rate']>summary['defect_rates']['validation']['observed_rate']
        assert summary['defect_rates']['train']['observed_rate']>summary['defect_rates']['test']['observed_rate']
    (root/'validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
