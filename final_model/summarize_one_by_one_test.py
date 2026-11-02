"""Summarize the completed one-vial-at-a-time test, including persistent misses."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'final_model/one-by-one-test-006';j=json.loads((out/'results.json').read_text());status=json.loads((out/'status.json').read_text());assert status['stage']=='COMPLETE' and j['completed_videos']==10
 histories=[]
 for row in j['per_video']:
  sid=row['sequence_id'];seq=json.loads((out/'sequences'/(sid+'.json')).read_text());preds=[json.loads(line) for line in (out/'inspection'/sid/'frame-results.jsonl').read_text().splitlines()]
  for slot,defect in row['defect_plan'].items():
   visible=raw=confirmed=cleared=0;ever=False;max_score=0.
   for frame,pred in zip(seq['frames'],preds):
    gt={a['slot_id'] for a in frame['annotations'] if a['category_id'] in (2,3)}
    if slot not in gt:continue
    visible+=1;found=any(d['slot_id']==slot for d in pred['raw_cap_defects']);raw+=found
    found_confirmed=any(d['slot_id']==slot for d in pred['cap_defects']);confirmed+=found_confirmed
    if ever and not found_confirmed:cleared+=1
    ever|=found_confirmed
    max_score=max(max_score,max((d['score'] for d in pred['raw_cap_defects'] if d['slot_id']==slot),default=0.))
   histories.append({'video':sid,'slot_id':slot,'kind':defect['kind'],'tilt_degrees':defect['tilt_degrees'],'visible_defect_frames':visible,'raw_detected_frames':raw,'confirmed_frames':confirmed,'ever_detected':raw>0,'ever_confirmed':ever,'confirmed_then_cleared_frames':cleared,'max_above_cutoff_score':max_score})
 with (out/'defect-histories.csv').open('w',newline='') as stream:
  w=csv.DictWriter(stream,fieldnames=list(histories[0]));w.writeheader();w.writerows(histories)
 a=j['aggregate'];raw=a['raw_cap'];confirmed=a['confirmed_cap'];settled=a['settled_cap'];order=a['order'];settled_order=a['settled_order']
 lines=['# One-vial-at-a-time synthetic testing: 006-final','','Ten new videos, 31 placement states per video, 155 frames each: 1,550 source frames and 1,550 annotated frames. Videos start empty and add exactly one vial per second, G01 through G30, at 5 FPS. Each is 31 seconds long. The placement sidecars verified exactly one new slot per state, no removals/gaps, and stable source-vial identity and cap condition. Both source and annotated videos were decoded to verify H.264 and all 155 frames.','','[Review the ten source and annotated videos](video-index.md). Annotated copies include independent Actual, raw Model and temporally Confirmed summaries.','','Two videos are clean controls. Eight independently sample persistent cap defects with probability 0.10 per placed vial. New seeds 6100000–6100009 were used; source geometry families and camera remain shared with the existing synthetic data. No weights were trained or changed.','','Settings: final compression-aware model rtdetr-v3-006-final.ts, cap cutoff 0.90, presence cutoff 0.25, confirmation/absence three frames. Source videos use H.264 (unlike the older mp4v benchmark inputs), so content, pacing and codec differ from that earlier test; this is not a controlled codec comparison.','','## Aggregate cap-defect results','','| Metric | Raw cap candidates | Confirmed caps, all frames | Confirmed caps, settled frames |','|---|---:|---:|---:|']
 lines.append('| Defect recall | '+' | '.join(f"{c['detected_defects']/c['defective_slot_frames']:.2%}" for c in (raw,confirmed,settled))+' |')
 for label,key in [('Detected defective slot-frames','detected_defects'),('Missed defective slot-frames','missed_defects'),('False alarms on good slots','false_alarms_good'),('False alarms on empty slots','false_alarms_empty')]:lines.append('| '+label+' | '+' | '.join(str(c[key]) for c in (raw,confirmed,settled))+' |')
 lines+=['','Settled means the last three frames of each five-frame placement state, omitting the first two frames for temporal confirmation. Counts are correlated slot-frames, not unique independent physical vials. A detection of either cap-defect subtype counts as finding a defective slot.','',f"Missing-vial alerts: {a['missing_vial']['false_missing_slot_frames']} false missing slot-frames; all sequences have no intentional gaps. This tests normal-fill specificity, not missing-vial recall.",'',f"Full-tray recognition: {order['correct_complete_frames']}/{order['actual_complete_frames']} full-tray frames, {order['false_complete_frames']} false-complete frames; after confirmation settles: {settled_order['correct_complete_frames']}/{settled_order['actual_complete_frames']}.",'','## Per-video results','','| Video | Labeled defective vials | Raw missed slot-frames | Raw false alarms on good slots | False missing-vial slot-frames |','|---|---:|---:|---:|---:|']
 for row in j['per_video']:
  c=row['metrics']['raw_cap'];lines.append(f"| [{row['sequence_id']}]({row['inspected_video']}) | {len(row['defect_plan'])} | {c['missed_defects']} | {c['false_alarms_good']} | {row['metrics']['missing_vial']['false_missing_slot_frames']} |")
 missed=[r for r in histories if not r['ever_detected']]
 lines+=['','## Defects never detected at cutoff 0.90','','| Video | Slot | Generator condition | Visible frames |','|---|---|---|']
 for row in missed:
  kind='missing cap' if row['kind']=='missing' else f"tilt {row['tilt_degrees']:.2f} degrees"
  lines.append(f"| {row['video']} | {row['slot_id']} | {kind} | {row['visible_defect_frames']} |")
 if not missed:lines.append('| None | | | |')
 lines+=['',f"Across the eight sampled-defect sequences, {sum(r['ever_detected'] for r in histories)}/{len(histories)} distinct video/slot defect instances were detected at least once, and {sum(r['ever_confirmed'] for r in histories)}/{len(histories)} were confirmed at least once. This is a separate measure from frame-level recall.",'','Defect-histories.csv records visible, detected and confirmed frames per defect, including confirmations later cleared while the defect persisted. Results.json contains per-video and aggregate counts; config.json records generation/runtime settings.','','Limitations: discrete static state changes rather than physical insertion motion; fixed synthetic camera; shared geometry families; only ten seeded videos; high cutoff may reject subtle defects. No real-camera performance claim or threshold tuning is made.']
 (out/'findings.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:25]))
if __name__=='__main__':main()
