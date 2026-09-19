from __future__ import annotations
import math,torch
from torch import nn
import torch.nn.functional as F
class SinusoidalTimeEmbedding(nn.Module):
 def __init__(self,dim): super().__init__(); self.dim=dim
 def forward(self,t):
  h=self.dim//2; f=torch.exp(-math.log(10000.0)*torch.arange(h,device=t.device)/max(h-1,1)); e=t.float()[:,None]*f[None,:]; return torch.cat([e.sin(),e.cos()],-1)
class VideoResBlock(nn.Module):
 def __init__(self,ic,oc,td,cd):
  super().__init__(); self.n1=nn.GroupNorm(8,ic); self.s1=nn.Conv3d(ic,oc,(1,3,3),padding=(0,1,1)); self.t1=nn.Conv3d(oc,oc,(3,1,1),padding=(1,0,0)); self.n2=nn.GroupNorm(8,oc); self.s2=nn.Conv3d(oc,oc,(1,3,3),padding=(0,1,1)); self.t2=nn.Conv3d(oc,oc,(3,1,1),padding=(1,0,0)); self.tp=nn.Linear(td,oc); self.cp=nn.Linear(cd,oc); self.skip=nn.Conv3d(ic,oc,1) if ic!=oc else nn.Identity()
 def forward(self,x,t,c):
  h=self.t1(self.s1(F.silu(self.n1(x)))); h=h+self.tp(t)[:,:,None,None,None]+self.cp(c)[:,:,None,None,None]; return self.t2(self.s2(F.silu(self.n2(h))))+self.skip(x)
class XENVideoModel(nn.Module):
 def __init__(self,cond_dim=384,base=32):
  super().__init__(); td=base*4; self.time=SinusoidalTimeEmbedding(td); self.cp=nn.Sequential(nn.Linear(cond_dim,cond_dim),nn.SiLU(),nn.Linear(cond_dim,cond_dim)); self.i=nn.Conv3d(3,base,3,padding=1); self.d1=VideoResBlock(base,base,td,cond_dim); self.d2=VideoResBlock(base,base*2,td,cond_dim); self.d3=VideoResBlock(base*2,base*4,td,cond_dim); self.m1=VideoResBlock(base*4,base*4,td,cond_dim); self.m2=VideoResBlock(base*4,base*4,td,cond_dim); self.u3=VideoResBlock(base*8,base*2,td,cond_dim); self.u2=VideoResBlock(base*4,base,td,cond_dim); self.u1=VideoResBlock(base*2,base,td,cond_dim); self.o=nn.Conv3d(base,3,3,padding=1)
 def forward(self,x,t,c):
  c=self.cp(c); t=self.time(t); x0=self.i(x); d1=self.d1(x0,t,c); d2=self.d2(F.avg_pool3d(d1,(1,2,2)),t,c); d3=self.d3(F.avg_pool3d(d2,(1,2,2)),t,c); m=self.m2(self.m1(d3,t,c),t,c); u3=self.u3(torch.cat([F.interpolate(m,size=d2.shape[-3:],mode='trilinear',align_corners=False),d2],1),t,c); u2=self.u2(torch.cat([F.interpolate(u3,size=d1.shape[-3:],mode='trilinear',align_corners=False),d1],1),t,c); return self.o(F.silu(self.u1(torch.cat([u2,x0],1),t,c)))
