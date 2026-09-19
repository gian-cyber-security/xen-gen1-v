#!/usr/bin/env python3
"""Synthetic TEST/SMOKE only for media model forward/backward/checkpoint I/O."""
from pathlib import Path
import sys, tempfile, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.tokenizer import XENTokenizer
if "image" in Path(__file__).resolve().parents[1].name:
    from model.image_model import XENImageModel
    from model.image_conditioner import XENImageTextEncoder
    kind = "image"
else:
    from model.video_model import XENVideoModel
    from model.video_conditioner import XENVideoTextEncoder
    kind = "video"

def main():
    torch.manual_seed(42); tok = XENTokenizer(); tok.fit(); ids = torch.tensor([tok.encode("TEST/SMOKE synthetic sample", 256 if kind == "image" else 512)])
    if kind == "image":
        model, cond, x = XENImageModel(), XENImageTextEncoder(), torch.randn(1, 3, 32, 32)
        t = torch.tensor([500]); noise = torch.randn_like(x); pred = model(x, t, cond(ids)); state = {"model": model.state_dict(), "conditioner": cond.state_dict(), "smoke_test": True}
    else:
        model, cond, x = XENVideoModel(), XENVideoTextEncoder(), torch.randn(1, 3, 2, 16, 16)
        t = torch.tensor([500]); noise = torch.randn_like(x); pred = model(x, t, cond(ids)); state = {"model": model.state_dict(), "conditioner": cond.state_dict(), "frames": 2, "size": 16, "smoke_test": True}
    loss = (pred - noise).pow(2).mean(); loss.backward(); opt = torch.optim.AdamW(list(model.parameters()) + list(cond.parameters()), lr=1e-4); opt.step()
    with tempfile.TemporaryDirectory(prefix="xen-smoke-") as d:
        p = Path(d); torch.save(state, p / "model.pt"); tok.save(p / "tokenizer.json"); z = torch.load(p / "model.pt", map_location="cpu", weights_only=False); assert z["smoke_test"]
    print("SMOKE TEST ONLY: PASS; no real dataset or training claim")
if __name__ == "__main__": main()
