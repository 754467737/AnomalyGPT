import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.anomalyany_synth_dataset import AnomalyAnySynthDataset
from models.anomalyany_clip_detector import AnomalyAnyCLIPDetector


def dice_loss(pred, target, eps=1e-6):
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


def save_vis(out_dir, batch, preds, limit=5):
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs, masks = batch[0], batch[1]
    for i in range(min(limit, imgs.size(0))):
        np.save(out_dir / f"sample_{i}_gt.npy", masks[i, 0].cpu().numpy())
        np.save(out_dir / f"sample_{i}_mvl.npy", preds["M_VL"][i, 0].detach().cpu().numpy())
        np.save(out_dir / f"sample_{i}_mvv.npy", preds["M_VV"][i, 0].detach().cpu().numpy())
        np.save(out_dir / f"sample_{i}_final.npy", preds["pixel_map"][i, 0].detach().cpu().numpy())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--obj_name", default=None, help="None means all objects under data_root")
    ap.add_argument("--seed", required=True, help="e.g. seed_0")
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--selected_layers", type=int, nargs="*", default=None)
    ap.add_argument("--lambda_img", type=float, default=1.0)
    ap.add_argument("--lambda_pix", type=float, default=1.0)
    ap.add_argument("--mask_threshold", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ds = AnomalyAnySynthDataset(data_root=args.data_root, obj_name=args.obj_name, seed=args.seed, image_size=args.image_size, mask_threshold=args.mask_threshold)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    normal_dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = AnomalyAnyCLIPDetector(args.clip_model, args.image_size, args.selected_layers, debug=args.debug).to(device)
    text_cache = {}

    opt = torch.optim.Adam(model.trainable_parameters, lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = 1e9; logs = []
    for epoch in range(args.epochs):
        model.rebuild_memory_bank(normal_dl, device)
        model.train(); total = 0.0
        for batch in dl:
            img, mask, label, _, objs, defects = batch
            img, mask, label = img.to(device), mask.to(device), label.to(device)
            # class-agnostic normal/anomaly prompts with defect-aware abnormal token
            f_text_list = []
            for obj, defect in zip(objs, defects):
                key = (obj, defect)
                if key not in text_cache:
                    text_cache[key] = model.build_text_features(obj, device, defect_word=defect)
                f_text_list.append(text_cache[key])
            # run sample-wise because prompts differ per sample
            outs = [model(img[i:i+1], f_text_list[i]) for i in range(img.size(0))]
            pred = {k: torch.cat([o[k] for o in outs], dim=0) for k in outs[0].keys()}

            assert pred["pixel_map"].shape[-2:] == mask.shape[-2:]
            img_loss = F.binary_cross_entropy(pred["image_score"].sigmoid(), label)
            pix_loss = F.binary_cross_entropy(pred["pixel_map"], mask) + dice_loss(pred["pixel_map"], mask)
            loss = args.lambda_img * img_loss + args.lambda_pix * pix_loss
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item())
        sch.step()

        avg = total / max(len(dl), 1)
        logs.append({"epoch": epoch, "loss": avg})
        torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "latest.pth")
        if avg < best:
            best = avg; torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "best.pth")
        if epoch % 10 == 0:
            save_vis(out_dir / "vis" / f"epoch_{epoch}", batch, pred, limit=5)

    (out_dir / "loss_log.json").write_text(json.dumps(logs, indent=2))
    with open(out_dir / "loss_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "loss"]); w.writeheader(); w.writerows(logs)
