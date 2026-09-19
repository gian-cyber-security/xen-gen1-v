import torch
from torch import nn
class XENVideoTextEncoder(nn.Module):
 def __init__(self,vocab_size=260,dim=384,max_tokens=512):
  super().__init__(); self.embedding=nn.Embedding(vocab_size,dim,padding_idx=0); self.position=nn.Embedding(max_tokens,dim); self.norm=nn.LayerNorm(dim)
 def forward(self,input_ids):
  input_ids=input_ids[:,:self.position.num_embeddings]; p=torch.arange(input_ids.shape[1],device=input_ids.device)[None,:]; x=self.embedding(input_ids)+self.position(p); m=(input_ids!=0).float()[:,:,None]; return self.norm((x*m).sum(1)/m.sum(1).clamp_min(1))
