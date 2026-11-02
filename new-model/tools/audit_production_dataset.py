"""Audit generated dataset integrity and publish conservative duplicate-filtered CSVs."""
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image,ImageDraw

root=Path('new-model/dataset-300')
qa=root/'qa-production';qa.mkdir(exist_ok=True)
records=[json.loads(x) for x in (root/'manifest.jsonl').read_text().splitlines()]
slots=defaultdict(list)
for r in records:slots[r['scene'],r['tray']].append(r)
for rows in slots.values():
 assert len(rows)==30 and {r['position'] for r in rows}==set(range(1,31))
 last=max([r['position'] for r in rows if r['present']],default=0)
 assert all(r['gap_error']==(not r['present'] and r['position']<last) for r in rows)
golden_hash=hashlib.sha256(Path('new-model/GoldenImages/G24.jpg').read_bytes()).hexdigest()
groups=defaultdict(list);issues=[];counts=Counter();families=defaultdict(set)
for r in records:
 if r['excluded']:continue
 sample=r['sample_id'];path=root/'images/inspection'/(sample+'_SolderLight.jpg')
 try:
  with Image.open(path) as im:
   im.load();assert im.mode=='RGB' and im.size==(90,108)
  ref=root/'images/golden'/path.name
  assert hashlib.sha256(ref.read_bytes()).hexdigest()==golden_hash
  a,b,c,d=r['inspection_bounds_native'];x0,y0,x1,y1=r['bbox']
  assert x0<=(a+.5)*3-.5 and x1>=(c+.5)*3-.5
  assert y0<=(b+.5)*3-.5-360 and y1>=(d+.5)*3-.5-360
  if r.get('label_source') == 'relative_cap_tilt_v1':
   tilt=r['cap_tilt_degrees']
   expected='PASS' if r['cap_present'] and tilt is not None and tilt < 1.0 and abs(tilt-1.0)>1e-9 else 'NO_PASS'
   assert r['label']==expected
  else:
   assert r['label']==('NO_PASS' if r['source_id'].startswith('S') else 'PASS')
 except Exception as e:issues.append(dict(sample_id=sample,reason=repr(e)));continue
 groups[hashlib.sha256(path.read_bytes()).hexdigest()].append(r)
 counts[r['split']+'/'+r['label']]+=1
 families[r['source_id']].add(r['split'])
# Exact copies carry no new information; retain train before validation before test.
rank={'train':0,'validation':1,'test':2};excluded={r['sample_id'] for r in issues}
duplicate_groups=[]
for digest,rows in groups.items():
 if len(rows)>1:
  ordered=sorted(rows,key=lambda r:(rank[r['split']],r['sample_id']))
  excluded.update(r['sample_id'] for r in ordered[1:])
  duplicate_groups.append(dict(hash=digest,keep=ordered[0]['sample_id'],exclude=[r['sample_id'] for r in ordered[1:]],splits=sorted({r['split'] for r in rows})))
reviewed=root/'csv-reviewed';reviewed.mkdir(exist_ok=True)
filtered_counts={}
for split in ('train','validation','test','inference'):
 with (root/'csv'/f'{split}.csv').open() as f:
  reader=csv.DictReader(f);header=reader.fieldnames;rows=[r for r in reader if r['object_name'] not in excluded]
 with (reviewed/f'{split}.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=header);writer.writeheader();writer.writerows(rows)
 filtered_counts[split]=dict(Counter(r['label'] for r in rows))
# Stratified source/split/tray coverage, deterministic spread across each stratum.
strata=defaultdict(list)
for r in records:
 if r.get('sample_id') and r['sample_id'] not in excluded:strata[r['split'],r['label'],r['tray']].append(r)
sampled=[]
for key,rows in sorted(strata.items()):
 rows.sort(key=lambda r:r['sample_id']);sampled.extend(rows[round(i*(len(rows)-1)/min(11,len(rows)-1))] for i in range(min(12,len(rows))))
for page in range((len(sampled)+47)//48):
 sheet=Image.new('RGB',(960,900),'white');draw=ImageDraw.Draw(sheet)
 for i,r in enumerate(sampled[page*48:(page+1)*48]):
  x=(i%8)*120;y=(i//8)*150
  with Image.open(root/'images/inspection'/(r['sample_id']+'_SolderLight.jpg')) as im:sheet.paste(im,(x,y))
  draw.text((x,y+109),r['sample_id'].replace('capseal_',''),fill='black');draw.text((x,y+122),r['label']+' '+r['split'],fill='black');draw.text((x,y+135),r['source_id'],fill='black')
 sheet.save(qa/f'stratified_{page+1}.jpg')
(qa/'visual_sample.json').write_text(json.dumps(sampled,indent=2)+'\n')
report=dict(slots_checked=len(records),pairs_checked=sum(counts.values()),issues=issues,exact_duplicate_groups=len(duplicate_groups),cross_split_duplicate_groups=sum(len(g['splits'])>1 for g in duplicate_groups),excluded_samples=sorted(excluded),filtered_counts=filtered_counts,source_family_overlap={k:sorted(v) for k,v in families.items() if len(v)>1},visual_samples=len(sampled),status='automated_audit_complete_visual_sample_pending',limitations=['Scene-disjoint splits share original assembly families; no independent-family generalization claim.','JPEG hash checks detect exact copies, not all perceptual near-duplicates.','Visual QA is stratified sampling, not exhaustive manual review.'])
(qa/'audit.json').write_text(json.dumps(report,indent=2)+'\n');(qa/'duplicates.json').write_text(json.dumps(duplicate_groups,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('source_family_overlap','excluded_samples')},indent=2))
