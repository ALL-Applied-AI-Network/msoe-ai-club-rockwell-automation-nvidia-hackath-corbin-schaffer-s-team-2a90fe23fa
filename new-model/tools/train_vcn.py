"""Native single-GPU Visual ChangeNet training from the verified smoke configuration."""
import json,math,os,traceback
from datetime import datetime,timezone
from pathlib import Path
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer,seed_everything,Callback
from pytorch_lightning.loggers import CSVLogger
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.pl_oi_data_module import OIDataModule
from nvidia_tao_pytorch.cv.visual_changenet.classification.models.cn_pl_model import ChangeNetPlModel

root=Path(__file__).resolve().parents[2]
out=root/'new-model/training-digital-001'
status=out/'run-status.json'
def write_status(stage,**kwargs):
 tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**kwargs),indent=2)+'\n');tmp.replace(status)
class Progress(Callback):
 def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
  loss=outputs['loss'] if isinstance(outputs,dict) else outputs
  if loss is not None and not torch.isfinite(loss).all():raise RuntimeError('Non-finite training loss')
  if trainer.global_step==1 or trainer.global_step%25==0:
   write_status('RUNNING',epoch=trainer.current_epoch+1,global_step=trainer.global_step,train_loss=float(loss.detach().cpu()),target_epochs=30)
 def on_validation_end(self,trainer,model):
  if trainer.sanity_checking:return
  metrics={k:float(v.detach().cpu()) for k,v in trainer.callback_metrics.items() if isinstance(v,torch.Tensor) and v.numel()==1}
  if not all(math.isfinite(v) for v in metrics.values()):raise RuntimeError('Non-finite validation metrics')
  write_status('RUNNING',epoch=trainer.current_epoch+1,global_step=trainer.global_step,metrics=metrics,target_epochs=30)
try:
 seed_everything(42,workers=True)
 cfg=OmegaConf.load(out/'spec.yaml')
 write_status('INITIALIZING',target_epochs=30)
 dm=OIDataModule(cfg,changenet=True);dm.setup('fit')
 model=ChangeNetPlModel(cfg,dm=dm)
 trainer=Trainer(accelerator='gpu',devices=1,max_epochs=cfg.train.num_epochs,
    check_val_every_n_epoch=1,logger=CSVLogger(str(out),name='metrics'),
    callbacks=[Progress()],enable_checkpointing=True,enable_progress_bar=False,
    default_root_dir=str(out),num_sanity_val_steps=2,precision='32-true',
    log_every_n_steps=25,use_distributed_sampler=False)
 trainer.fit(model,datamodule=dm)
 best_callbacks=[callback for callback in trainer.checkpoint_callbacks
                 if getattr(callback,'monitor',None)=='val_loss' and getattr(callback,'best_model_path','')]
 best_path=best_callbacks[0].best_model_path if best_callbacks else None
 write_status('COMPLETE',epochs=trainer.current_epoch,global_step=trainer.global_step,
              best_checkpoint=best_path)
except BaseException:
 write_status('ERROR',traceback=traceback.format_exc());raise
