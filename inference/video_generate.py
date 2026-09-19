import argparse,math
from pathlib import Path
import imageio.v3 as iio,numpy as np,torch
from model.tokenizer import XENTokenizer
from model.video_conditioner import XENVideoTextEncoder
from model.video_model import XENVideoModel
def ab(t): return torch.cos(((t.float()/999)+.008)/1.008*math.pi/2).pow(2).clamp(1e-4,.9999)
@torch.no_grad()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model-dir',default='outputs/xen-gen1-v'); p.add_argument('--prompt',required=True); p.add_argument('--output',default='outputs/xen-gen1-v/generated.mp4'); p.add_argument('--duration',type=float,default=2); p.add_argument('--frames',type=int); p.add_argument('--size',type=int,default=128); p.add_argument('--steps',type=int,default=50); p.add_argument('--seed',type=int,default=42); p.add_argument('--fps',type=int,default=8); a=p.parse_args();
 if not 1<=a.duration<=15: raise ValueError('--duration must be between 1 and 15 seconds'); frames=a.frames or max(2,round(a.duration*a.fps)); d=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); z=torch.load(Path(a.model_dir)/'model.pt',map_location=d,weights_only=False); tok=XENTokenizer.load(Path(a.model_dir)/'tokenizer.json'); m=XENVideoModel().to(d); c=XENVideoTextEncoder().to(d); m.load_state_dict(z['model']); c.load_state_dict(z['conditioner']); m.eval(); c.eval(); ids=torch.tensor([tok.encode(a.prompt,max_length=512)],device=d); cond=c(ids); g=torch.Generator(device=d).manual_seed(a.seed); x=torch.randn((1,3,frames,a.size,a.size),device=d,generator=g); ts=torch.linspace(999,0,a.steps,device=d)
 for i in range(len(ts)-1):
  t=ts[i].expand(1); tn=ts[i+1].expand(1); at=ab(t)[:,None,None,None,None]; an=ab(tn)[:,None,None,None,None]; e=m(x,t,cond); x0=((x-(1-at).sqrt()*e)/at.sqrt().clamp_min(1e-4)).clamp(-1.5,1.5); x=an.sqrt()*x0+(1-an).sqrt()*e
 arr=((x[0].clamp(-1,1)+1)*127.5).byte().permute(1,2,3,0).cpu().numpy(); Path(a.output).parent.mkdir(parents=True,exist_ok=True); iio.imwrite(a.output,arr,plugin='ffmpeg',fps=a.fps); print('Saved video:',a.output)
if __name__=='__main__': main()
