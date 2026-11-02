"""Native RT-DETR on dataset-v3 keyframes; smoke and training are separate invocations."""
import argparse,json,math,os,traceback
from pathlib import Path
from datetime import datetime,timezone
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer,Callback,seed_everything
from pytorch_lightning.loggers import CSVLogger
from nvidia_tao_pytorch.config.rtdetr.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.rtdetr.dataloader.pl_od_data_module import ODDataModule
from nvidia_tao_pytorch.cv.rtdetr.model.pl_rtdetr_model import RTDETRPlModel

p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true')
p.add_argument('--output',type=Path)
p.add_argument('--resume',type=Path)
p.add_argument('--additional-epochs',type=int,default=50)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--dataset-root',type=Path)
p.add_argument('--max-train-batches',type=int)
p.add_argument('--paired-validation',action='store_true')
args=p.parse_args()
if args.additional_epochs<1:p.error('additional-epochs must be positive')
root=Path(__file__).resolve().parents[1]
out=args.output.resolve() if args.output else root/'final_model/training-v3-001';out=out/'smoke' if args.smoke else out;out.mkdir(parents=True,exist_ok=True)
base_epochs=0;target_epochs=50;initial_global_step=0
status=out/'run-status.json'
def state(stage,**kw):
 tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**kw),indent=2)+'\n');tmp.replace(status)
class Progress(Callback):
 def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
  loss=outputs['loss'] if isinstance(outputs,dict) else outputs
  if loss is not None and not torch.isfinite(loss).all():raise RuntimeError('Non-finite loss')
  if trainer.global_step==1 or trainer.global_step%10==0:state('RUNNING',epoch=trainer.current_epoch+1,global_step=trainer.global_step,loss=float(loss.detach().cpu()),target_epochs=target_epochs,additional_epochs_completed=max(0,trainer.current_epoch+1-base_epochs))
 def on_validation_end(self,trainer,model):
  if trainer.sanity_checking:return
  metrics={k:float(v.detach().cpu()) for k,v in trainer.callback_metrics.items() if isinstance(v,torch.Tensor) and v.numel()==1}
  if not all(math.isfinite(v) for v in metrics.values()):raise RuntimeError('Non-finite validation metrics')
  state('RUNNING',epoch=trainer.current_epoch+1,global_step=trainer.global_step,metrics=metrics,target_epochs=target_epochs,additional_epochs_completed=max(0,trainer.current_epoch+1-base_epochs))
try:
 if args.resume:
  args.resume=args.resume.resolve()
  saved=torch.load(args.resume,map_location='cpu',weights_only=False)
  if not saved.get('optimizer_states') or not saved.get('lr_schedulers'):raise ValueError('Resume checkpoint lacks optimizer or scheduler state')
  base_epochs=int(saved['epoch'])+1;initial_global_step=int(saved['global_step']);target_epochs=base_epochs+args.additional_epochs
  # Validation-end checkpoints precede epoch-end hooks. Mark a fully processed
  # train epoch complete so Lightning does not replay hooks with empty loss state.
  fit=saved.get('loops',{}).get('fit_loop',{})
  batch=fit.get('epoch_loop.batch_progress',{})
  progress=fit.get('epoch_progress',{})
  current=batch.get('current',{})
  if batch.get('is_last_batch') and current.get('ready',0)>0 and current.get('completed')==current.get('ready') and progress.get('current',{}).get('completed',base_epochs)<base_epochs:
   for bucket in ('current','total'):
    for key in ('ready','started','processed','completed'):progress[bucket][key]=base_epochs
   def clear_current(node):
    if not isinstance(node,dict):return
    for key,value in node.items():
     if key=='current' and isinstance(value,dict):
      for name in value:value[name]=0
     elif key=='is_last_batch':node[key]=False
     else:clear_current(value)
   for name,node in fit.items():
    if name!='epoch_progress':clear_current(node)
   normalized=out/'resume_epoch_boundary.pth'
   torch.save(saved,normalized)
   (out/'resume-normalization.json').write_text(json.dumps({'source':str(args.resume),'copy':str(normalized),'completed_epochs':base_epochs,'global_step':initial_global_step,'change':'Only loop boundary progress adjusted; tensors, optimizer and scheduler retained.'},indent=2)+'\n')
   args.resume=normalized
  del saved
 seed_everything(args.seed,workers=True);state('INITIALIZING',smoke=args.smoke,seed=args.seed,resume_from=str(args.resume) if args.resume else None,base_epochs=base_epochs,target_epochs=target_epochs)
 dataset=args.dataset_root.resolve() if args.dataset_root else root/'final_model/dataset-v3'
 cfg=OmegaConf.structured(ExperimentConfig)
 overrides={'results_dir':str(out),'wandb':{'enable':False},
  'model':{'backbone':'resnet_50','pretrained_backbone_path':str(root/'final_model/models/resnet50-imagenet.pth'),
           'train_backbone':True,'return_interm_indices':[1,2,3]},
  'train':{'seed':args.seed,'num_epochs':target_epochs,'resume_training_checkpoint_path':str(args.resume) if args.resume else None,'num_gpus':1,'gpu_ids':[0],'num_nodes':1,'validation_interval':1,'checkpoint_interval':1,
           'precision':'fp32','enable_ema':False,'activation_checkpoint':False,
           'optim':{'lr':.0001,'lr_backbone':.00001,'lr_steps':[35,45]},
           'checkpointer':{'enable_topk':True,'replace_periodic':True,'monitor':'val_mAP50','mode':'max','save_top_k':1}},
  'dataset':{'dataset_type':'default','batch_size':4,'workers':0 if args.smoke else 2,'num_classes':4,
             'eval_class_ids':[1,2,3],'remap_mscoco_category':False,
             'train_data_sources':[{'image_dir':str(dataset/'frames'),'json_file':str(dataset/'annotations/train-keyframes.json')}],
             'val_data_sources':{'image_dir':str(dataset/'frames'),'json_file':str(dataset/'annotations/validation-keyframes.json')},
             'augmentation':{'multi_scales':[640],'train_spatial_size':[640,640],'eval_spatial_size':[640,640],
                             'distortion_prob':.2,'iou_crop_prob':0.0,'preserve_aspect_ratio':False}}}
 cfg=OmegaConf.merge(cfg,overrides);OmegaConf.save(cfg,out/'spec.yaml')
 dm=ODDataModule(cfg.dataset);dm.setup('fit')
 model=RTDETRPlModel(cfg)
 callbacks=[Progress()]
 if args.paired_validation:
  from compression_validation import CompressionValidation
  callbacks.append(CompressionValidation(dataset,out,int(cfg.model.num_select),target_epochs))
 trainer=Trainer(accelerator='gpu',devices=1,max_epochs=base_epochs+1 if args.smoke else target_epochs,
     max_steps=initial_global_step+1 if args.smoke else -1,
     fast_dev_run=1 if args.smoke and not args.resume else False,
     logger=False if args.smoke else CSVLogger(str(out),name='metrics'),callbacks=callbacks,
     enable_checkpointing=not args.smoke,enable_progress_bar=False,default_root_dir=str(out),
     limit_train_batches=1 if args.smoke else (args.max_train_batches if args.max_train_batches else 1.0),
     limit_val_batches=1 if args.smoke else 1.0,
     num_sanity_val_steps=0 if args.smoke else 2,precision='32-true',gradient_clip_val=.1,
     check_val_every_n_epoch=1,log_every_n_steps=10,use_distributed_sampler=False)
 trainer.fit(model,datamodule=dm,ckpt_path=str(args.resume) if args.resume else None)
 if args.smoke:assert trainer.global_step==initial_global_step+1
 final_checkpoint=None
 if not args.smoke:
  final_checkpoint=out/f'final_epoch_{trainer.current_epoch:03d}.pth'
  trainer.save_checkpoint(str(final_checkpoint))
 best=[c for c in trainer.checkpoint_callbacks if getattr(c,'monitor',None)=='val_mAP50' and getattr(c,'best_model_path','')]
 state('SMOKE_PASSED' if args.smoke else 'COMPLETE',epochs=trainer.current_epoch,global_step=trainer.global_step,
       best_checkpoint=best[0].best_model_path if best else None,final_checkpoint=str(final_checkpoint) if final_checkpoint else None,
       resume_from=str(args.resume) if args.resume else None,base_epochs=base_epochs,additional_epochs=trainer.current_epoch-base_epochs,max_memory_allocated_bytes=torch.cuda.max_memory_allocated())
except BaseException:
 state('ERROR',traceback=traceback.format_exc());raise
