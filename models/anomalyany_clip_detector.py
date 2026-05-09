import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


def _minmax_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


class AnomalyAnyCLIPDetector(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch16", image_size=224, selected_layers=None, prompt_templates=None, use_shared_adapter=True, debug=False):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.image_size = image_size
        self.debug = debug
        self.prompt_templates = list(prompt_templates) if prompt_templates else ["a photo of a {} {}."]
        self.selected_layers = list(selected_layers or ([3, 6, 9, 12] if "base" in clip_model_name else [6, 12, 18, 24]))

        for p in self.clip.parameters():
            p.requires_grad = False
        assert not any(p.requires_grad for p in self.clip.parameters())

        vh = self.clip.config.vision_config.hidden_size
        th = self.clip.config.projection_dim
        self.shared_adapter = use_shared_adapter
        self.adapter = nn.Linear(vh, th) if use_shared_adapter else None
        self.adapters = None if use_shared_adapter else nn.ModuleList([nn.Linear(vh, th) for _ in self.selected_layers])
        self.memory_banks: Dict[str, Dict[int, torch.Tensor]] = {}

    @property
    def trainable_parameters(self):
        return self.adapter.parameters() if self.shared_adapter else self.adapters.parameters()

    def build_text_features(self, class_name: str, device: torch.device, defect_word: str = "anomalous") -> torch.Tensor:
        normal = [t.format("normal", class_name) for t in self.prompt_templates]
        abnormal = [t.format(defect_word, class_name) for t in self.prompt_templates]
        inputs = self.processor(text=normal + abnormal, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_feats = F.normalize(self.clip.get_text_features(**inputs), dim=-1)
        n = len(self.prompt_templates)
        return F.normalize(torch.stack([text_feats[:n].mean(0), text_feats[n:].mean(0)], dim=0), dim=-1)

    def _extract_vision(self, images):
        out = self.clip.vision_model(pixel_values=images, output_hidden_states=True)
        f_image = F.normalize(self.clip.visual_projection(out.pooler_output), dim=-1)
        patches = []
        for l in self.selected_layers:
            tok = out.hidden_states[l][:, 1:, :]
            b, n, _ = tok.shape
            s = int(math.sqrt(n)); assert s * s == n
            patches.append((tok, s, s))
        return f_image, patches

    def _adapt_patch(self, tokens, layer_i):
        x = self.adapter(tokens) if self.shared_adapter else self.adapters[layer_i](tokens)
        return F.normalize(x, dim=-1)

    @torch.no_grad()
    def rebuild_memory_bank(self, normal_loader, device: torch.device):
        self.eval()
        per_obj = {}
        for images, _, labels, _, objs, _ in normal_loader:
            images = images.to(device)
            labels = labels.to(device)
            keep = labels == 0
            if keep.sum() == 0:
                continue
            images = images[keep]
            objs = [o for o, k in zip(objs, keep.tolist()) if k]
            _, patches = self._extract_vision(images)
            for j, obj in enumerate(objs):
                per_obj.setdefault(obj, {i: [] for i in range(len(self.selected_layers))})
                for i, (tok, _, _) in enumerate(patches):
                    feat = self._adapt_patch(tok[j:j+1], i).reshape(-1, self.clip.config.projection_dim)
                    per_obj[obj][i].append(feat)
        self.memory_banks = {}
        for obj, d in per_obj.items():
            self.memory_banks[obj] = {}
            for i, lst in d.items():
                assert len(lst) > 0, f"Empty memory bank: obj={obj}, layer={i}"
                self.memory_banks[obj][i] = torch.cat(lst, 0).detach().to(device)

    def forward(self, images: torch.Tensor, f_text: torch.Tensor, obj_names):
        b, _, h, w = images.shape
        f_image, patches = self._extract_vision(images)
        # f_text: [B,2,C] or [2,C]
        if f_text.dim() == 2:
            f_text = f_text.unsqueeze(0).expand(b, -1, -1)

        vl_logits = torch.einsum("bc,bkc->bk", f_image, f_text)
        s_vl = torch.softmax(vl_logits, dim=-1)
        s_vl_abn = s_vl[:, 1]

        m_vl = torch.zeros((b, 1, h, w), device=images.device)
        m_vv = torch.zeros((b, 1, h, w), device=images.device)

        for i, (tok, ph, pw) in enumerate(patches):
            adapted = self._adapt_patch(tok, i)
            logits = torch.einsum("bnc,bkc->bnk", adapted, f_text)
            probs = torch.softmax(logits, dim=-1)[..., 1].reshape(b, 1, ph, pw)
            m_vl += F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=False)

            for bi, obj in enumerate(obj_names):
                bank = self.memory_banks.get(obj, {}).get(i)
                assert bank is not None and bank.numel() > 0, f"Missing memory bank for obj={obj}, layer={i}"
                sim = adapted[bi] @ bank.t()
                vv = 1.0 - sim.max(dim=-1).values.reshape(1, 1, ph, pw)
                m_vv[bi:bi+1] += F.interpolate(vv, size=(h, w), mode="bilinear", align_corners=False)

        m_vl = _minmax_norm(m_vl); m_vv = _minmax_norm(m_vv)
        s_vv = m_vv.flatten(1).max(-1).values
        final_pixel = _minmax_norm(m_vl + m_vv)
        if self.debug:
            print(f"[DEBUG] F_image={f_image.shape}, F_text={f_text.shape}, M_VL={m_vl.shape}, M_VV={m_vv.shape}")
        return {"image_score": s_vl_abn + s_vv, "pixel_map": final_pixel, "S_VL": s_vl, "M_VL": m_vl, "M_VV": m_vv, "S_VV": s_vv}
