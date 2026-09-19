import argparse,json,random
from pathlib import Path
import imageio.v3 as iio,numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from model.tokenizer import XENTokenizer
from model.video_conditioner import XENVideoTextEncoder
from model.video_model import XENVideoModel
class DS(Dataset):
 def __init__(self,p,t,frames,size):
  self.rows=[(x['video'],x['caption']) for x in map(json.loads,open(p,encoding='utf-8')) if x]; self.t=t; self.f=frames; self.s=size
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  path,cap=self.rows[i]; frames=list(iio.imiter(path,plugin='ffmpeg')); arr=np.asarray(frames); n=len(arr); start=0 if n<self.f else random.randint(0,n-self.f); idx=np.linspace(0,n-1,self.f).round().astype(int) if n<self.f else np.arange(start,start+self.f); clip=torch.from_numpy(arr[idx]).permute(3,0,1,2).float()/127.5-1; clip=F.interpolate(clip[None],size=(self.f,self.s,self.s),mode='trilinear',align_corners=False)[0]; return clip,torch.tensor(self.t.encode(cap,max_length=512))
def collate(b):
 m=max(x[1].numel() for x in b); return torch.stack([x[0] for x in b]),torch.stack([F.pad(x[1],(0,m-x[1].numel())) for x in b])
def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',default='datasets/video_data.jsonl'); p.add_argument('--output',default='outputs/xen-gen1-v'); p.add_argument('--frames',type=int,default=16); p.add_argument('--size',type=int,default=128); p.add_argument('--batch-size',type=int,default=1); p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--steps',type=int,default=10000); p.add_argument('--grad-accumulation',type=int,default=8); p.add_argument('--resume',type=str); a=p.parse_args()
 if a.grad_accumulation<1: raise ValueError('--grad-accumulation must be >= 1')
 t=XENTokenizer(); t.fit(); dl=DataLoader(DS(a.data,t,a.frames,a.size),batch_size=a.batch_size,shuffle=True,collate_fn=collate); d=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=XENVideoModel().to(d); c=XENVideoTextEncoder().to(d); ps=list(m.parameters())+list(c.parameters()); o=torch.optim.AdamW(ps,lr=a.lr,weight_decay=.01); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); t.save(out/'tokenizer.json'); step=0; o.zero_grad(set_to_none=True)
 if a.resume:
  z=torch.load(a.resume,map_location=d,weights_only=False); m.load_state_dict(z['model']); c.load_state_dict(z['conditioner']); o.load_state_dict(z['optimizer']); step=int(z.get('step',0)); print(f'resumed step={step}')
 while step<a.steps:
  for micro,(v,ids) in enumerate(dl):
   v,ids=v.to(d),ids.to(d); tt=torch.randint(0,1000,(v.shape[0],),device=d); n=torch.randn_like(v); ab=torch.cos(((tt.float()/999)+.008)/1.008*torch.pi/2).pow(2).clamp(1e-4,.9999)[:,None,None,None,None]; pred=m(ab.sqrt()*v+(1-ab).sqrt()*n,tt,c(ids)); loss=F.mse_loss(pred,n); (loss/a.grad_accumulation).backward()
   if (micro+1)%a.grad_accumulation==0:
    torch.nn.utils.clip_grad_norm_(ps,1); o.step(); o.zero_grad(set_to_none=True); step+=1
    if step%20==0: print(f'step={step} loss={loss.item():.5f}')
    if step%500==0: torch.save({'model':m.state_dict(),'conditioner':c.state_dict(),'optimizer':o.state_dict(),'step':step,'frames':a.frames,'size':a.size,'grad_accumulation':a.grad_accumulation},out/f'checkpoint-{step}.pt')
    if step>=a.steps: break
  if step>=a.steps: break
 torch.save({'model':m.state_dict(),'conditioner':c.state_dict(),'optimizer':o.state_dict(),'step':step,'frames':a.frames,'size':a.size,'grad_accumulation':a.grad_accumulation},out/'model.pt')
if __name__=='__main__': main()
