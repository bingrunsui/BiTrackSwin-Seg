from __future__ import annotations
import argparse, numpy as np, torch
from pathlib import Path
from PIL import Image
from common import read_config, model_from_config
from datasets import read_raster
from engine import load_model_checkpoint

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--input",required=True); p.add_argument("--output",required=True); p.add_argument("--device",default="cuda"); a=p.parse_args(); c=read_config(a.config); d=torch.device(a.device if torch.cuda.is_available() else "cpu")
    image=read_raster(a.input).astype(np.float32)
    if image.shape[-1] != c["model"]["input_channels"]: raise ValueError("input must be a five-band raster")
    norm=c["data"]["normalization"]; image=np.clip((image-np.asarray(norm["mean"],np.float32))/(np.asarray(norm["std"],np.float32)+1e-8), *norm["clip"])
    x=torch.from_numpy(image.transpose(2,0,1)).unsqueeze(0); x=torch.nn.functional.interpolate(x, tuple(c["model"]["image_size"]), mode="bilinear",align_corners=False).to(d)
    m=model_from_config(c).to(d).eval(); load_model_checkpoint(m,a.checkpoint,d,strict=True)
    with torch.inference_mode(): mask=m(x).argmax(1)[0].byte().cpu().numpy()*255
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Image.fromarray(mask).save(a.output)
if __name__=="__main__": main()
