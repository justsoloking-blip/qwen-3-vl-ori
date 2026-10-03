# 中文对照阅读版。保留原始实验逻辑，未修复positions stride缺陷。
# 该候选未进入最终默认方案；请结合PDF阅读，不能视为已验收通用内核。
import os
"""Experimental Qwen3-VL BF16 Q/K RMSNorm + interleaved MRoPE.
Narrow contract: head_dim=rotary_dim=128, Neox layout, sections [24,20,20].
Outputs remain separate Q/K tensors; this does not fuse the KV cache write.
"""
import torch
import triton
import triton.language as tl

# 第10行：JIT编译GPU内核。第27行包装函数负责分配输出与launch。
@triton.jit
# N是运行时token数量；NQ/NK是每rank Q/K头数。
# QS/KS是输入token行stride；EPS是Norm稳定项；这些constexpr可形成编译变体。
def _qk_norm_mrope(Q,K,QW,KW,POS,CACHE,OQ,OK, N, NQ:tl.constexpr,NK:tl.constexpr,QS:tl.constexpr,KS:tl.constexpr, EPS:tl.constexpr):
    # 每个program=(token,head)。h是head，不是height；d是64个旋转对。
    t=tl.program_id(0);h=tl.program_id(1);d=tl.arange(0,64)
    # 前NQ个program-head选择Q，其余选择K。两类head独立归一化。
    if h<NQ:
        # 输入用实际QS寻址，输出OQ连续；指针偏移均以元素计。
        src=Q+t*QS+h*128;w=QW;dst=OQ+t*NQ*128+h*128
    else:
        # K指针本身已指向K视图，局部head编号是h-NQ。
        src=K+t*KS+(h-NQ)*128;w=KW;dst=OK+t*NK*128+(h-NQ)*128
    # NeoX配对是(d,d+64)；加载两半128维，转FP32做Norm。
    x0=tl.load(src+d).to(tl.float32);x1=tl.load(src+d+64).to(tl.float32)
    # 一个head内求128维均方和倒平方根，不减均值，不跨head/token归约。
    inv=tl.rsqrt(tl.sum(x0*x0+x1*x1,axis=0)/128+EPS)
    # QW/KW各128个权重，对该head的两个半区按维缩放。
    w0=tl.load(w+d).to(tl.float32);w1=tl.load(w+d+64).to(tl.float32)
    # Preserve the BF16 intermediate rounding of the original RMSNorm.
    # 显式BF16舍入，靠近原Norm输出BF16再做RoPE的路径；并非逐位保证。
    a=(x0*inv*w0).to(Q.dtype.element_ty);b=(x1*inv*w1).to(Q.dtype.element_ty)
    # 轴分配：d0..59为[T,H,W]重复20次，d60..63全T。总计24T/20H/20W。
    ax=tl.where((d%3==1)&(d<=60),1,tl.where((d%3==2)&(d<=60),2,0))
    # 已知错误：ax*N+t假定positions连续stride=(N,1)。
    # [3,17]缓冲区切成[3,4]时真实stride仍(17,1)，该公式会读错。
    pos=tl.load(POS+ax*N+t)
    # 每个cache行前64个cos，后64个sin；按选中轴的位置和频率d查表。
    c=tl.load(CACHE+pos*128+d);s=tl.load(CACHE+pos*128+d+64)
    # 二维旋转并写回d/d+64。接入时a/b/c/s为BF16，不能称整段FP32。
    tl.store(dst+d,a*c-b*s);tl.store(dst+d+64,b*c+a*s)

# Python包装函数；输出保持Q/K独立，未融合KV Cache写入。
def fused_qk_norm_mrope(q,k,qweight,kweight,positions,cache,eps=1e-6):
    # 新输出按shape分配为连续张量，输入仍可为QKV split行stride视图。
    oq=torch.empty(q.shape,device=q.device,dtype=q.dtype);ok=torch.empty(k.shape,device=k.device,dtype=k.dtype)
    # grid=(N,NQ+NK)，例如TP2为(N,36)。num_warps不是program个数。
    # enable_fp_fusion=False限制乘加融合，不保证参考逐位相同。
    _qk_norm_mrope[(q.shape[0],q.shape[1]//128+k.shape[1]//128)](q,k,qweight,kweight,positions,cache,oq,ok,q.shape[0],q.shape[1]//128,k.shape[1]//128,q.stride(0),k.stride(0),eps,num_warps=4,enable_fp_fusion=False)
    return oq,ok
