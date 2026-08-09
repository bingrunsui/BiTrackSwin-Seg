"""Reference training entry point for the released single-stream v1 model."""
from __future__ import annotations
import argparse, torch
from torch.utils.data import DataLoader
from common import read_config, model_from_config
from datasets import FiveBandSegmentationDataset

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--output",default="outputs/v1_last.pth"); p.add_argument("--epochs",type=int); p.add_argument("--device",default="cuda"); p.add_argument("--dry-run",action="store_true"); p.add_argument("--smoke",action="store_true",help="run the intentionally simplified smoke loop"); a=p.parse_args(); c=read_config(a.config); d=torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(c["runtime"]["seed"])
    model=model_from_config(c).to(d)
    if a.dry_run:
        print(f"model-ready parameters={sum(p.numel() for p in model.parameters())}")
        return
    if not a.smoke:
        raise RuntimeError("Full v1 reproduction is not implemented in this public entry yet: the original loss/augmentation/EMA/WSD pipeline must not be silently replaced by CE. Use --smoke only to validate an installation.")
    data=c["data"]; ds=FiveBandSegmentationDataset(data["root"],data["train_manifest"],data["image_dir"],data["label_dir"],data["normalization"],c["model"]["image_size"],c["model"].get("ignore_index",255),data.get("nodata_threshold",-999.0))
    loader=DataLoader(ds,batch_size=c["train"]["batch_size"],shuffle=True,num_workers=c["train"].get("num_workers",0),pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=c["train"]["optimizer"]["lr"],weight_decay=c["train"]["optimizer"]["weight_decay"]); weights=torch.tensor(c["train"]["loss"]["class_weights"],device=d)
    for epoch in range(a.epochs or c["train"]["epochs"]):
        model.train(); total=0.
        for batch in loader:
            logits=model(batch["image"].to(d)); loss=torch.nn.functional.cross_entropy(logits,batch["label"].to(d),weight=weights,ignore_index=c["model"].get("ignore_index",255)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); total+=loss.item()
        print(f"epoch={epoch + 1} loss={total / max(len(loader),1):.6f}")
    torch.save({"model_state_dict":model.state_dict(),"config":c},a.output)
if __name__=="__main__": main()
