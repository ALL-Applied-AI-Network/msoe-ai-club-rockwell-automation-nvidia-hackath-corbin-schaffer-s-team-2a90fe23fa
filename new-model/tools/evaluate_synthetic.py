"""Fixed-threshold, single-pass evaluation on the untouched synthetic test split."""
import csv,json,math,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score,average_precision_score
from PIL import Image,ImageDraw
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.pl_oi_data_module import OIDataModule
from nvidia_tao_pytorch.cv.visual_changenet.classification.models.cn_pl_model import ChangeNetPlModel
root=Path(__file__).resolve().parents[2]
out=root/'new-model/evaluation-synthetic-001';status=out/'status.json'
def state(stage,**kw):
 temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**kw),indent=2)+'\n');temp.replace(status)
try:
 started=time.monotonic();state('INITIALIZING')
 cfg=OmegaConf.load(root/'new-model/training-digital-001/spec.yaml')
 cfg.results_dir=str(out);cfg.wandb.enable=False
 dataset=root/'new-model/dataset-300-tilt'
 cfg.dataset.classify.infer_dataset.csv_path=str(dataset/'csv/test.csv')
 cfg.dataset.classify.infer_dataset.images_dir=str(dataset/'images')
 # Full state is in the trained checkpoint; avoid redundant backbone initialization load.
 cfg.model.backbone.pretrained_backbone_path=''
 OmegaConf.save(cfg,out/'spec.yaml')
 dm=OIDataModule(cfg,changenet=True);dm.setup('predict');loader=dm.predict_dataloader()
 checkpoint=root/'new-model/training-digital-001/model_best_028.pth'
 model=ChangeNetPlModel(cfg,dm=dm)
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
 model.load_state_dict(saved['state_dict'],strict=True)
 del saved
 model=model.cuda().eval()
 threshold=float(cfg.model.classify.eval_margin);records=[]
 with torch.inference_mode():
  for index,batch in enumerate(loader):
   batch=[x.cuda(non_blocking=True) for x in batch]
   scores=model._forward_pass(batch).cpu().tolist()
   labels=batch[2].view(-1).long().cpu().tolist()
   for score,label in zip(scores,labels):
    assert math.isfinite(score)
    source=dm.df_infer.iloc[len(records)]
    assert label==int(source['label']!='PASS')
    pred=int(score>threshold)
    records.append(dict(sample_id=str(source['object_name']),actual='NO_PASS' if label else 'PASS',
                        predicted='NO_PASS' if pred else 'PASS',defect_score=score,
                        correct=pred==label,outcome=('TP' if pred else 'FN') if label else ('FP' if pred else 'TN')))
   if index%10==0:state('RUNNING',processed=len(records),total=len(dm.df_infer))
 assert len(records)==len(dm.df_infer)==2285
 assert len({r['sample_id'] for r in records})==len(records)
 counts={key:sum(r['outcome']==key for r in records) for key in ['TP','TN','FP','FN']}
 def ratio(n,d):return n/d if d else None
 tp,tn,fp,fn=(counts[k] for k in ['TP','TN','FP','FN'])
 metrics={'accuracy':ratio(tp+tn,len(records)),'defect_recall':ratio(tp,tp+fn),'defect_precision':ratio(tp,tp+fp),
          'false_positive_rate':ratio(fp,fp+tn),'false_negative_rate':ratio(fn,fn+tp),
          'defect_f1':ratio(2*tp,2*tp+fp+fn),
          'roc_auc':roc_auc_score([int(r['actual']=='NO_PASS') for r in records],[r['defect_score'] for r in records]),
          'average_precision':average_precision_score([int(r['actual']=='NO_PASS') for r in records],[r['defect_score'] for r in records])}
 for name,rows in [('predictions',records),('failures',[r for r in records if not r['correct']])]:
  with (out/f'{name}.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(rows)
 failures=[r for r in records if not r['correct']]
 if failures:
  sheet=Image.new('RGB',(800,150*math.ceil(min(len(failures),80)/8)),'white');draw=ImageDraw.Draw(sheet)
  for i,r in enumerate(failures[:80]):
   x=(i%8)*100;y=(i//8)*150
   with Image.open(dataset/'images/inspection'/(r['sample_id']+'_SolderLight.jpg')) as image:sheet.paste(image,(x,y))
   draw.text((x,y+109),r['sample_id'].replace('capseal_',''),fill='black')
   draw.text((x,y+122),r['outcome']+' '+format(r['defect_score'],'.3f'),fill='black')
  sheet.save(out/'failure-contact-sheet.jpg')
 result=dict(stage='COMPLETE',samples=len(records),threshold=threshold,checkpoint=str(checkpoint),
             confusion_matrix=counts,metrics=metrics,elapsed_seconds=time.monotonic()-started,
             limitations=['Synthetic rendered scenes share source vial families with training.',
                          'Threshold fixed at 0.5 before testing; no test-set calibration.',
                          'Real-camera performance not evaluated.'])
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Held-out synthetic evaluation','',f'Evaluated {len(records):,} test pairs once, at fixed defect-score threshold {threshold}.',
        '', '| Outcome | Count |','|---|---:|']
 for k,v in counts.items():lines.append(f'| {k} | {v} |')
 lines+=['','TP/FN refer to defective vials; TN/FP refer to good vials.','','| Metric | Value |','|---|---:|']
 for k,v in metrics.items():lines.append(f'| {k} | {v:.6f} |' if v is not None else f'| {k} | Undefined |')
 lines+=['','Rates are fractions (multiply by 100 for percentages). AUC and average precision are ranking metrics.',
         '', 'Checkpoint: `model_best_028.pth` (chosen by training validation loss).',
         '', 'This measures unseen synthetic scenes, not unseen source families or real-camera accuracy.',
         '', 'See [predictions.csv](predictions.csv), [failures.csv](failures.csv), and [results.json](results.json).']
 (out/'evaluation-statistics.md').write_text('\n'.join(lines)+'\n')
 state('COMPLETE',samples=len(records),confusion_matrix=counts,metrics=metrics)
 print(json.dumps(result,indent=2))
except BaseException:
 state('ERROR',traceback=traceback.format_exc());raise
