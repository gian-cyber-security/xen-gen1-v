import argparse,math
from pathlib import Path
import imageio.v3 as iio,numpy as np,torch
from PIL import Image
from model.tokenizer import XENTokenizer
from model.video_conditioner import XENVideoTextEncoder
from model.video_model import XENVideoModel
from tools.prompt_search import enrich_prompt
def ab(t): return torch.cos(((t.float()/999)+.008)/1.008*math.pi/2).pow(2).clamp(1e-4,.9999)
def load_image(path,size,device):
 im=Image.open(path).convert("RGB").resize((size,size))
 arr=np.asarray(im)
 return torch.from_numpy(arr).permute(2,0,1).float().div(127.5).sub(1).to(device)
@torch.no_grad()
def main():
 p=argparse.ArgumentParser(description="XEN-GEN1-V text-to-video and image-to-video")
 p.add_argument("--model-dir",default="outputs/xen-gen1-v"); p.add_argument("--prompt",required=True)
 p.add_argument("--image",help="Reference image for image-to-video generation")
 p.add_argument("--strength",type=float,default=0.65,help="Motion/edit strength, lower preserves the reference more")
 p.add_argument("--output",default="outputs/xen-gen1-v/generated.mp4"); p.add_argument("--duration",type=float,default=2)
 p.add_argument("--frames",type=int); p.add_argument("--size",type=int,default=128); p.add_argument("--steps",type=int,default=50)
 p.add_argument("--seed",type=int,default=42); p.add_argument("--fps",type=int,default=8); p.add_argument("--web-search",action="store_true"); p.add_argument("--no-web-search",action="store_true"); a=p.parse_args()
 if a.web_search and not a.no_web_search: a.prompt=enrich_prompt(a.prompt)
 if not 1<=a.duration<=15: raise ValueError("--duration must be between 1 and 15 seconds")
 if not 0.05<=a.strength<=1.0: raise ValueError("--strength must be between 0.05 and 1.0")
 frames=a.frames or max(2,round(a.duration*a.fps))
 d=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 z=torch.load(Path(a.model_dir)/"model.pt",map_location=d,weights_only=False)
 tok=XENTokenizer.load(Path(a.model_dir)/"tokenizer.json")
 m=XENVideoModel().to(d); c=XENVideoTextEncoder().to(d); m.load_state_dict(z["model"]); c.load_state_dict(z["conditioner"]); m.eval(); c.eval()
 ids=torch.tensor([tok.encode(a.prompt,max_length=512)],device=d); cond=c(ids)
 g=torch.Generator(device=d).manual_seed(a.seed)
 if a.image:
  ref=load_image(a.image,a.size,d).unsqueeze(1).repeat(1,frames,1,1,1)
  start=max(1,min(999,round(999*a.strength))); ts=torch.linspace(start,0,a.steps,device=d)
  st=torch.tensor([start],device=d); x=ab(st).sqrt()[:,None,None,None,None]*ref+(1-ab(st)).sqrt()[:,None,None,None,None]*torch.randn(ref.shape,device=d,generator=g)
 else:
  ts=torch.linspace(999,0,a.steps,device=d); x=torch.randn((1,3,frames,a.size,a.size),device=d,generator=g)
 for i in range(len(ts)-1):
  t=ts[i].expand(1); tn=ts[i+1].expand(1); at=ab(t)[:,None,None,None,None]; an=ab(tn)[:,None,None,None,None]
  e=m(x,t,cond); x0=((x-(1-at).sqrt()*e)/at.sqrt().clamp_min(1e-4)).clamp(-1.5,1.5); x=an.sqrt()*x0+(1-an).sqrt()*e
  if a.image and i < len(ts)-2:
   ref_t=an.sqrt()*ref+(1-an).sqrt()*torch.zeros_like(ref)
   x[:, :, :1] = ref_t[:, :, :1]
 arr=((x[0].clamp(-1,1)+1)*127.5).byte().permute(1,2,3,0).cpu().numpy()
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); iio.imwrite(a.output,arr,plugin="ffmpeg",fps=a.fps); print("Saved video:",a.output)
if __name__=="__main__": main()
