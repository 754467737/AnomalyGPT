import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.anomalyany_synth_dataset import AnomalyAnySynthDataset, MVTecTestDataset
from models.anomalyany_clip_detector import AnomalyAnyCLIPDetector
from utils.anomaly_metrics import max_f1, safe_ap, safe_auroc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--seed", required=True, help="e.g. seed_0")
    ap.add_argument("--class_name", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--dataset_type", default="mvtec", choices=["mvtec", "visa"])
    ap.add_argument("--clip_model", default="ViT-L/14")
    ap.add_argument("--clip_download_root", default="~/.cache/clip")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--selected_layers", type=int, nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    model = AnomalyAnyCLIPDetector(clip_model_name=args.clip_model, clip_download_root=args.clip_download_root, image_size=args.image_size, selected_layers=args.selected_layers, debug=args.debug).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    normal_ds = AnomalyAnySynthDataset(data_root=args.data_root, obj_name=args.class_name, seed=args.seed, image_size=args.image_size)
    normal_dl = DataLoader(normal_ds, batch_size=16, shuffle=False, num_workers=args.num_workers)
    model.rebuild_memory_bank(normal_dl, device)
    f_text = model.build_text_features(args.class_name, device)

    test_ds = MVTecTestDataset(args.dataset_root, args.class_name, args.image_size)
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=args.num_workers)

    y_img, s_img = [], []
    y_pix, s_pix = [], []
    with torch.no_grad():
        for images, masks, labels, paths in test_dl:
            images = images.to(device)
            obj_names = [args.class_name] * images.size(0)
            pred = model(images, f_text, obj_names)
            scores = pred["image_score"].detach().cpu().numpy()
            maps = pred["pixel_map"].detach().cpu().numpy()
            y_img.extend(labels.numpy().tolist()); s_img.extend(scores.tolist())
            y_pix.extend(masks.numpy().reshape(-1).tolist()); s_pix.extend(maps.reshape(-1).tolist())
            for i, p in enumerate(paths):
                np.save(out / (Path(p).stem + "_anomaly.npy"), maps[i, 0])

    metrics = {
        "image_auroc": safe_auroc(y_img, s_img),
        "image_ap": safe_ap(y_img, s_img),
        "image_max_f1": max_f1(y_img, s_img),
        "pixel_auroc": safe_auroc(y_pix, s_pix),
        "pixel_ap": safe_ap(y_pix, s_pix),
        "pixel_max_f1": max_f1(y_pix, s_pix),
        "pro": None,
        "pro_todo": "TODO: add exact PRO metric implementation if required by benchmark protocol.",
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
