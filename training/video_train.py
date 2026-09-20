from __future__ import annotations
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
        self.rows=[(x["video"],x["caption"]) for x in map(json.loads,open(p,encoding="utf-8")) if x]
        if not self.rows: raise ValueError("Dataset is empty")
        self.t=t; self.f=frames; self.s=size
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        path,cap=self.rows[i]; arr=np.asarray(list(iio.imiter(path,plugin="ffmpeg")))
        if arr.ndim!=4 or arr.shape[-1]<3: raise ValueError(f"Invalid video: {path}")
        n=len(arr)
        if n<self.f: idx=np.linspace(0,n-1,self.f).round().astype(int)
        else:
            start=random.randint(0,n-self.f); idx=np.arange(start,start+self.f)
        clip=torch.from_numpy(arr[idx,:,:,:3]).permute(3,0,1,2).float()/127.5-1
        clip=F.interpolate(clip[None],size=(self.f,self.s,self.s),mode="trilinear",align_corners=False)[0]
        return clip,torch.tensor(self.t.encode(cap,max_length=512),dtype=torch.long)

def collate(b):
    m=max(x[1].numel() for x in b)
    return torch.stack([x[0] for x in b]),torch.stack([F.pad(x[1],(0,m-x[1].numel())) for x in b])

def update_ema(ema,model,decay):
    with torch.no_grad():
        for e,p in zip(ema.parameters(),model.parameters()):
            e.mul_(decay).add_(p,alpha=1-decay)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="datasets/video_data.jsonl"); p.add_argument("--output",default="outputs/xen-gen1-v")
    p.add_argument("--frames",type=int,default=16); p.add_argument("--size",type=int,default=128)
    p.add_argument("--batch-size",type=int,default=1); p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--steps",type=int,default=20000); p.add_argument("--grad-accumulation",type=int,default=8)
    p.add_argument("--save-every",type=int,default=250); p.add_argument("--resume",default=None); p.add_argument("--ema-decay",type=float,default=.999)
    a=p.parse_args()
    random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    t=XENTokenizer(); t.fit()
    dl=DataLoader(DS(a.data,t,a.frames,a.size),batch_size=a.batch_size,shuffle=True,collate_fn=collate)
    d=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m=XENVideoModel().to(d); c=XENVideoTextEncoder().to(d)
    ema=XENVideoModel().to(d); ema.load_state_dict(m.state_dict()); ema.eval()
    ps=list(m.parameters())+list(c.parameters()); opt=torch.optim.AdamW(ps,lr=a.lr,weight_decay=.01,betas=(.9,.99))
    scaler=torch.amp.GradScaler("cuda",enabled=d.type=="cuda")
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); t.save(out/"tokenizer.json"); step=0; it=iter(dl)
    if a.resume:
        ck=torch.load(a.resume,map_location=d,weights_only=False)
        m.load_state_dict(ck["model"]); c.load_state_dict(ck["conditioner"])
        if "ema" in ck: ema.load_state_dict(ck["ema"])
        if "optimizer" in ck: opt.load_state_dict(ck["optimizer"])
        if "scaler" in ck and d.type=="cuda": scaler.load_state_dict(ck["scaler"])
        step=int(ck.get("step",0)); print(f"Resumed from step={step}")
    m.train(); c.train(); accum=max(1,a.grad_accumulation)
    while step<a.steps:
        opt.zero_grad(set_to_none=True); total=0.0
        for _ in range(accum):
            try: clip,ids=next(it)
            except StopIteration: it=iter(dl); clip,ids=next(it)
            clip,ids=clip.to(d,non_blocking=True),ids.to(d,non_blocking=True)
            tt=torch.randint(0,1000,(clip.shape[0],),device=d)
            ab=torch.cos(((tt.float()/999)+.008)/1.008*torch.pi/2).pow(2).clamp(1e-4,.9999)[:,None,None,None,None]
            with torch.autocast(device_type=d.type,dtype=torch.float16,enabled=d.type=="cuda"):
                noise=torch.randn_like(clip); noisy=ab.sqrt()*clip+(1-ab).sqrt()*noise
                pred=m(noisy,tt,c(ids)); loss=F.mse_loss(pred,noise)/accum
            scaler.scale(loss).backward(); total+=loss.item()*accum
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(ps,1.0); scaler.step(opt); scaler.update()
        update_ema(ema,m,a.ema_decay); step+=1
        if step%10==0: print(f"step={step} loss={total/accum:.5f}")
        if step%a.save_every==0:
            torch.save({"model":m.state_dict(),"ema":ema.state_dict(),"conditioner":c.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"step":step,"frames":a.frames,"size":a.size},out/f"checkpoint-{step}.pt")
            print(f"checkpoint saved: step={step}")
    torch.save({"model":ema.state_dict(),"conditioner":c.state_dict(),"optimizer":opt.state_dict(),"step":step,"frames":a.frames,"size":a.size},out/"model.pt")
    print(f"model saved: {out/'model.pt'}")

if __name__=="__main__": main()
