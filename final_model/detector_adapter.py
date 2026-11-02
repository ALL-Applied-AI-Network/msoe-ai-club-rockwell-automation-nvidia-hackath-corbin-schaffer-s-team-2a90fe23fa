"""RT-DETR checkpoint adapter for native-frame video inference."""
import torch
from torch import nn
from torch.nn import functional as F

class RTDETRVideoAdapter(nn.Module):
    def __init__(self,detector,eval_size=(640,640),num_select=300,native_size=(480,640)):
        super().__init__()
        self.detector=detector
        self.eval_size=eval_size
        self.num_select=num_select
        self.native_size=native_size
    def forward(self,image):
        # Match eval resize + uint8-to-float handling; RT-DETR has no ImageNet normalization.
        resized=F.interpolate(image,size=self.eval_size,mode='bilinear',align_corners=False,antialias=True)
        resized=torch.round(resized.clamp(0,1)*255.0)/255.0
        outputs=self.detector(resized)
        logits,boxes=outputs['pred_logits'],outputs['pred_boxes']
        scores,indices=torch.topk(logits.sigmoid().flatten(1),self.num_select,dim=1)
        labels=indices%logits.shape[-1]
        queries=torch.div(indices,logits.shape[-1],rounding_mode='floor')
        xyxy=torch.cat((boxes[...,:2]-boxes[...,2:]/2,boxes[...,:2]+boxes[...,2:]/2),dim=-1)
        xyxy=torch.gather(xyxy,1,queries.unsqueeze(-1).expand(-1,-1,4))
        scale=xyxy.new_tensor([self.native_size[1],self.native_size[0],self.native_size[1],self.native_size[0]])
        xyxy=xyxy*scale
        # Category zero is unused by this trained head. Keep known categories only.
        valid=(labels[0]>=1)&(labels[0]<=3)
        return torch.cat((xyxy[0],scores[0].unsqueeze(1),labels[0].to(xyxy.dtype).unsqueeze(1)),dim=1)[valid]

class VideoInputContract(nn.Module):
    """Scripted shape checks outside the fixed-shape traced detector graph."""
    def __init__(self,model):
        super().__init__();self.model=model
    def forward(self,image:torch.Tensor)->torch.Tensor:
        if image.dim()!=4 or image.size(0)!=1 or image.size(1)!=3 or image.size(2)!=480 or image.size(3)!=640:
            raise RuntimeError('Expected one RGB frame: float32 [1,3,480,640]')
        if image.dtype!=torch.float32:
            raise RuntimeError('Expected float32 RGB values in [0,1]')
        return self.model(image)

class ExportAttention(nn.Module):
    """Use TAO's registered inference CUDA op instead of a Python autograd call."""
    def __init__(self,attention):
        super().__init__();self.attention=attention
    def forward(self,query,reference_points,values,spatial_shapes,level_starts,padding_mask=None):
        return self.attention(query,reference_points,values,spatial_shapes,level_starts,padding_mask,export=True)
