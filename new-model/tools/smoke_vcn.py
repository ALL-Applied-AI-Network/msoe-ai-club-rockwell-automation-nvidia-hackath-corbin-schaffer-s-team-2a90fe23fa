"""One train/validation batch on the accepted dataset; not a full training run."""
import json
import math
import traceback
from pathlib import Path
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from nvidia_tao_pytorch.config.visual_changenet.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.pl_oi_data_module import OIDataModule
from nvidia_tao_pytorch.cv.visual_changenet.classification.models.cn_pl_model import ChangeNetPlModel

root=Path(__file__).resolve().parents[2]
out=root/'new-model/smoke-test'
out.mkdir(exist_ok=True)
try:
 seed_everything(42,workers=True)
 cfg=OmegaConf.structured(ExperimentConfig)
 overrides={'task':'classify','results_dir':str(out),'wandb':{'enable':False},
 'model':{'backbone':{'type':'c_radio_v2_vit_base_patch16_224','pretrained_backbone_path':str(root/'new-model/base-model/model.safetensors')},
          'decode_head':{'feature_strides':[4,8,16,32],'use_summary_token':True},
          'classify':{'difference_module':'learnable','eval_margin':0.5,'embed_dec':30}},
 'train':{'num_epochs':1,'num_gpus':1,'gpu_ids':[0],'num_nodes':1,'validation_interval':1,'checkpoint_interval':1,
          'classify':{'loss':'ce'},'tensorboard':{'enabled':False},'use_distributed_sampler':False,'sync_batchnorm':False},
 'dataset':{'classify':{'batch_size':2,'workers':0,'image_ext':'.jpg','num_input':1,'input_map':{'SolderLight':0},
                      'image_width':224,'image_height':224,'augmentation_config':{'augment':False}}}}
 cfg=OmegaConf.merge(cfg,overrides)
 # Replace the default multi-light map, rather than merging its extra light entries.
 cfg.dataset.classify.input_map={'SolderLight':0}
 dataset=root/'new-model/dataset-300-tilt'
 for role,split in [('train_dataset','train'),('validation_dataset','validation')]:
  cfg.dataset.classify[role].csv_path=str(dataset/'csv'/f'{split}.csv')
  cfg.dataset.classify[role].images_dir=str(dataset/'images')
 OmegaConf.save(cfg,out/'spec.yaml')
 assert torch.cuda.is_available()
 dm=OIDataModule(cfg,changenet=True)
 dm.setup('fit')
 model=ChangeNetPlModel(cfg,dm=dm)
 trainer=Trainer(accelerator='gpu',devices=1,fast_dev_run=1,logger=False,
                 enable_checkpointing=False,enable_progress_bar=False,default_root_dir=str(out),
                 num_sanity_val_steps=0,precision='32-true')
 trainer.fit(model,datamodule=dm)
 metrics={k:float(v.detach().cpu()) if isinstance(v,torch.Tensor) else str(v) for k,v in trainer.callback_metrics.items()}
 assert all(math.isfinite(v) for v in metrics.values() if isinstance(v,float)), metrics
 result={'status':'passed','global_step':trainer.global_step,'gpu':torch.cuda.get_device_name(0),
         'torch':torch.__version__,'metrics':metrics,'max_memory_allocated_bytes':torch.cuda.max_memory_allocated()}
 assert trainer.global_step==1
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
except BaseException:
 (out/'result.json').write_text(json.dumps({'status':'error','traceback':traceback.format_exc()},indent=2)+'\n')
 raise
