#!/usr/bin/env python3
"""Export the trained RT-DETR checkpoint to inspect_video.py-compatible TorchScript."""
import argparse,hashlib,json,sys,shutil,sysconfig
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from detector_adapter import RTDETRVideoAdapter,VideoInputContract,ExportAttention
from nvidia_tao_pytorch.cv.rtdetr.model.pl_rtdetr_model import RTDETRPlModel
from nvidia_tao_pytorch.cv.rtdetr.model.postprocess import RTDETRPostProcess

def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=root/'final_model/training-v3-001/model_best_026.pth')
    p.add_argument('--spec',type=Path,default=root/'final_model/training-v3-001/spec.yaml')
    p.add_argument('--output',type=Path,default=root/'final_model/models/rtdetr-v3.ts')
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.output.exists():p.error('Output exists; choose a new filename')
    cfg=OmegaConf.load(args.spec)
    if cfg.dataset.augmentation.preserve_aspect_ratio:raise ValueError('This adapter supports the trained direct-resize pipeline, not letterboxing')
    if cfg.dataset.remap_mscoco_category:raise ValueError('Expected custom categories 1,2,3')
    cfg.model.pretrained_backbone_path=''
    model=RTDETRPlModel(cfg)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
    detector=model.model.to(args.device).eval()
    adapter=RTDETRVideoAdapter(detector,tuple(cfg.dataset.augmentation.eval_spatial_size),int(cfg.model.num_select)).eval().to(args.device)
    dataset=root/'final_model/dataset-v3'
    images=json.loads((dataset/'annotations/validation-keyframes.json').read_text())['images']
    indices=sorted(set([0,len(images)//3,2*len(images)//3,len(images)-1]))
    samples=[]
    for index in indices:
        with Image.open(dataset/'frames'/images[index]['file_name']) as im:
            array=np.asarray(im.convert('RGB')).copy()
        samples.append(torch.from_numpy(array).permute(2,0,1).unsqueeze(0).float().to(args.device)/255)
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        baselines=[adapter(sample).clone() for sample in samples]
        from nvidia_tao_pytorch.cv.deformable_detr.model.ops.modules import MSDeformAttn
        for parent in list(detector.modules()):
            for name,child in list(parent.named_children()):
                if isinstance(child,MSDeformAttn):setattr(parent,name,ExportAttention(child))
        # The architecture uses fixed eval position embeddings; intentionally trace native batch-one shape.
        traced=torch.jit.trace(adapter,samples[-1],strict=True,check_trace=False)
        scripted=torch.jit.script(VideoInputContract(traced).eval())
        errors=[];postprocess=RTDETRPostProcess(int(cfg.model.num_select),False)
        for sample,reference in zip(samples,baselines):
            actual=scripted(sample)
            torch.testing.assert_close(actual,reference,rtol=1e-4,atol=1e-3)
            errors.append(float((actual-reference).abs().max()) if actual.numel() else 0)
            # Independently verify adapter decoding against TAO's native postprocessor.
            resized=torch.nn.functional.interpolate(sample,size=tuple(cfg.dataset.augmentation.eval_spatial_size),mode='bilinear',align_corners=False,antialias=True)
            resized=torch.round(resized.clamp(0,1)*255)/255
            native=postprocess(detector(resized),torch.tensor([[480,640]],device=args.device),['frame'])[0]
            valid=(native['labels']>=1)&(native['labels']<=3)
            expected=torch.cat([native['boxes'],native['scores'][:,None],native['labels'].float()[:,None]],dim=1)[valid]
            torch.testing.assert_close(reference,expected,rtol=1e-5,atol=1e-4)
        try:scripted(torch.zeros((1,3,640,640),device=args.device))
        except (RuntimeError,torch.jit.Error):pass
        else:raise AssertionError('Export accepted an incorrect frame shape')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.tmp');torch.jit.save(scripted,str(temporary))
    loaded=torch.jit.load(str(temporary),map_location=args.device)
    with torch.inference_mode(), torch.jit.optimized_execution(False):torch.testing.assert_close(loaded(samples[-1]),adapter(samples[-1]),rtol=1e-4,atol=1e-3)
    temporary.replace(args.output)
    import nvidia_tao_pytorch
    ops_source=Path(nvidia_tao_pytorch.__file__).parent/'cv/deformable_detr/model/ops'/('MultiScaleDeformableAttention'+sysconfig.get_config_var('EXT_SUFFIX'))
    ops_target=args.output.with_suffix('.ops.so');shutil.copy2(ops_source,ops_target)
    metadata={'status':'exported_and_verified','checkpoint':str(args.checkpoint.resolve()),
              'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'artifact':str(args.output.resolve()),'required_libraries':[{'file':ops_target.name,'sha256':hashlib.sha256(ops_target.read_bytes()).hexdigest()}],
              'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'disable_jit_optimization':True,'supported_device':'cuda',
              'input_shape':[1,3,480,640],'input_dtype':'float32',
              'input_color':'RGB','input_range':[0,1],'network_size':list(cfg.dataset.augmentation.eval_spatial_size),
              'preprocessing':'antialiased bilinear resize; quantize to uint8 grid; scale [0,1]; no mean/std normalization',
              'output':'Nx6: native-frame x0,y0,x1,y1,score,category_id',
              'categories':{'1':'vial','2':'tilted_cap','3':'missing_cap'},
              'parity_validation_frames':len(samples),'maximum_absolute_trace_errors':errors,
              'notes':['No threshold filtering; inspect_video.py applies its runtime threshold.',
                       'Fixed 640x480 video, batch 1; CUDA extension/PyTorch ABI must match; real-camera calibration still required.',
                       'Resize approximates PIL training resize to within quantization; trace parity compares identical tensor preprocessing.']}
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
