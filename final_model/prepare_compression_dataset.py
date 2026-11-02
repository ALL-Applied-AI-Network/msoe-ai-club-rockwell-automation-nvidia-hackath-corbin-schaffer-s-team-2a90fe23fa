"""Build paired clean/decoded keyframes without changing sequence splits or labels."""
import copy,json,hashlib
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1]
def main():
 source=ROOT/'final_model/dataset-v3';out=ROOT/'final_model/dataset-v3-compression'
 out.mkdir(exist_ok=False);(out/'frames').mkdir();(out/'annotations').mkdir()
 (out/'frames/clean').symlink_to(source/'frames',target_is_directory=True)
 sets={};sequences={};counts={}
 for split in ('train','validation'):
  coco=json.loads((source/f'annotations/{split}-keyframes.json').read_text())
  by_seq={}
  for im in coco['images']:by_seq.setdefault(im['sequence_id'],{})[im['frame_index']]=im
  sequences[split]=set(by_seq)
  for seq,needed in by_seq.items():
   cap=cv2.VideoCapture(str(source/f'videos/{seq}.mp4'));assert cap.isOpened()
   index=0;found=set()
   while True:
    ok,frame=cap.read()
    if not ok:break
    if index in needed:
     im=needed[index];assert frame.shape[:2]==(im['height'],im['width'])
     target=out/'frames/compressed'/im['file_name'];target.parent.mkdir(parents=True,exist_ok=True)
     assert cv2.imwrite(str(target),frame);found.add(index)
    index+=1
   cap.release();assert found==set(needed)
  variants={}
  for variant in ('clean','compressed'):
   v=copy.deepcopy(coco)
   for im in v['images']:im['file_name']=variant+'/'+im['file_name'];im['compression_variant']=variant
   variants[variant]=v
   (out/f'annotations/{split}-{variant}.json').write_text(json.dumps(v))
  if split=='train':
   mixed=copy.deepcopy(variants['clean']);max_im=max(im['id'] for im in mixed['images']);max_ann=max(a['id'] for a in mixed['annotations'])
   compressed=variants['compressed']
   for im in compressed['images']:im['id']+=max_im
   for a in compressed['annotations']:a['id']+=max_ann;a['image_id']+=max_im
   mixed['images']+=compressed['images'];mixed['annotations']+=compressed['annotations']
   ids={im['id'] for im in mixed['images']};assert len(ids)==len(mixed['images'])
   assert len({a['id'] for a in mixed['annotations']})==len(mixed['annotations'])
   assert all(a['image_id'] in ids for a in mixed['annotations'])
   (out/'annotations/train-keyframes.json').write_text(json.dumps(mixed))
   sets[split]=mixed
  else:
   (out/'annotations/validation-keyframes.json').write_text(json.dumps(variants['clean']))
   sets[split]=variants['clean']
  counts[split]={'clean':len(coco['images']),'compressed':len(coco['images']),'sequence_count':len(by_seq)}
 assert not sequences['train']&sequences['validation']
 for coco in sets.values():
  for im in coco['images']:assert (out/'frames'/im['file_name']).is_file()
 (out/'calibration.json').symlink_to(source/'calibration.json')
 summary={'stage':'COMPLETE','source':str(source),'codec':'Existing OpenCV mp4v videos decoded with OpenCV, saved losslessly as PNG','counts':counts,'training_mix':'Equal clean/compressed candidate counts; shuffle and limit training to 99 batches per epoch to match clean-only update budget.','validation':'Clean COCO metrics each epoch; separate clean/compressed slot metrics periodically.','test':'Not decoded or included in this preparation.','annotation_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'annotations').glob('*.json')}}
 (out/'status.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
