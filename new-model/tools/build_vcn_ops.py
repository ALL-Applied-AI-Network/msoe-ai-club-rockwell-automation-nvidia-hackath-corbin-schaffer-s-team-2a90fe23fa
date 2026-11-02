"""Build only the TAO CUDA operator required by Visual ChangeNet."""
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension
root=Path(__file__).resolve().parents[2]
ops=root/'new-model/vendor/tao-pytorch/nvidia_tao_pytorch/cv/deformable_detr/model/ops'
setup(name='vcn-native-ops',ext_modules=[CUDAExtension(
    name='MultiScaleDeformableAttention',
    sources=[str(ops/'src'/name) for name in ['ms_deform_attn_cpu.cpp','ms_deform_attn_api.cpp','ms_deform_attn_cuda.cu']],
    include_dirs=[str(ops/'src')],define_macros=[('WITH_CUDA',None)],
    extra_compile_args={'cxx':[],'nvcc':['-DCUDA_HAS_FP16=1','-DCUTE_ARCH_MMA_SM90A_ENABLED=1',
        '-D__CUDA_NO_HALF_OPERATORS__','-D__CUDA_NO_HALF_CONVERSIONS__','-D__CUDA_NO_HALF2_OPERATORS__']} )],
    cmdclass={'build_ext':BuildExtension})
